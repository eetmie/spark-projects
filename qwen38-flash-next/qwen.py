"""Manual Qwen3.8 Flash-Next inference through an authenticated API."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('QWEN_FLASH_STATE', ROOT/'.local')).expanduser().resolve()
RECIPE = ROOT/'recipe'
IMAGE = 'ghcr.io/salient-data/qwen38-next-dgx@sha256:d55a2740651064f6c8aba20bc9ada104ae9df0a793cc87d0c170d312aa052b4d'
MODEL_REVISION = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'
import common


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def same_process(pid, script, role=None):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
        return state != 'Z' and any(
            Path(os.fsdecode(arg)).resolve() == script.resolve() and
            (role is None or index+1 < len(args) and os.fsdecode(args[index+1]) == role)
            for index, arg in enumerate(args) if arg)
    except (OSError, ValueError):
        return False


def active():
    current = read_json(STATE/'active.json')
    if current.get('pid') and same_process(current['pid'], ROOT/'qwen.py', role='worker'):
        return current
    return {}


def api(path):
    address = active().get('host', common.proxy_host())
    if address == '0.0.0.0':
        address = '127.0.0.1'
    req = urllib.request.Request(f'http://{address}:30001'+path,
        headers={'Authorization': 'Bearer '+common.key()})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=3) as response:
        return json.load(response)


def set_arg(args, flag, value):
    args = list(args)
    index = args.index(flag)
    args[index+1] = value
    return args


def salient_command():
    args = [line.strip() for line in (RECIPE/'serve.args').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    env = dict(line.split('=', 1) for line in (RECIPE/'serve.env').read_text().splitlines()
               if line.strip() and not line.startswith('#'))
    env['QWEN38_FP8_DENSE'] = ''
    args = set_arg(args, '--kv-cache-dtype', 'bfloat16')
    args = set_arg(args, '--mamba-ssm-cache-dtype', 'float32')
    args = set_arg(args, '--max-num-seqs', '2')
    args = set_arg(args, '--default-chat-template-kwargs', '{"enable_thinking":true,"reasoning_effort":"xhigh"}')
    index = args.index('--kv-cache-dtype-skip-layers')
    del args[index:index+2]
    data = STATE
    hf = data/'huggingface'
    command = ['docker', 'run', '--rm', '--init', '--name', 'qwen-salient-manual',
        '--gpus', 'all', '--network', 'host', '--ipc=host', '--shm-size', '32g',
        '--ulimit', 'memlock=-1:-1', '--memory', '112g', '--memory-swap', '112g',
        '-v', f'{hf}:/hf:ro', '-v', f'{data/"cache"}:/root/.cache']
    for name, value in env.items():
        command += ['-e', f'{name}={value}']
    # Bearer key is inherited by Docker by name; it never appears in argv.
    command += ['-e', 'VLLM_API_KEY', '--entrypoint', 'vllm', IMAGE, 'serve',
        f'/hf/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/{MODEL_REVISION}',
        '--host', '127.0.0.1', '--port', '30000', *args]
    return command


def verify_preparation():
    prepared = read_json(STATE/'preparation.json')
    if not prepared.get('passed') or prepared.get('stage') != 'prepared':
        raise RuntimeError('model is not prepared yet; run ./qwen prepare')
    snapshot = STATE/'huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots'/MODEL_REVISION
    if not snapshot.is_dir() or not read_json(STATE/'huggingface/verified-model.json').get('files'):
        raise RuntimeError('verified checkpoint is missing; rerun ./qwen prepare')
    # Recent Hub clients deduplicate some blobs outside the per-model directory.
    # Mount the whole HF_HOME, and refuse links that would escape even that mount.
    hf = (STATE/'huggingface').resolve()
    for file in snapshot.iterdir():
        if not file.is_file() or not file.resolve().is_relative_to(hf):
            raise RuntimeError(f'checkpoint file is not readable within the HF cache: {file.name}')
    subprocess.run(['docker', 'image', 'inspect', IMAGE], check=True, stdout=subprocess.DEVNULL)


def serve_salient():
    import serve
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, serve.stopped)
    with (STATE/'engine.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_preparation()
        address = common.preflight()
        if subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/qwen-salient-manual$'], text=True).strip():
            raise RuntimeError('qwen-salient-manual already exists; inspect it before retrying')
        cache = STATE/'cache/draft-token-maps'
        cache.mkdir(parents=True, exist_ok=True)
        for source in (RECIPE/'draft-token-maps').glob('*.json'):
            if hashlib.sha256(source.read_bytes()).hexdigest() != source.stem:
                raise RuntimeError('draft token map hash mismatch')
            target = cache/source.name
            if not target.exists() or target.read_bytes() != source.read_bytes():
                target.write_bytes(source.read_bytes())
        os.environ.update(UPSTREAM='http://127.0.0.1:30000', PROXY_BIND=address,
            VLLM_API_KEY=common.key(), PROMPT_CEILING_TOKENS='205000',
            FLASH_PROMPT_CEILING_TOKENS='205000', KEEPALIVE_S='10', MAX_SILENCE_S='3600',
            QWEN_FLASH_NATIVE_EFFORT_ALIASES='1')
        for name in ('NO_PROXY', 'no_proxy'):
            os.environ[name] = ','.join(filter(None, [os.environ.get(name), '127.0.0.1', 'localhost', address]))

        def cleanup():
            subprocess.run(['docker', 'stop', '-t', '20', 'qwen-salient-manual'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            subprocess.run(['docker', 'rm', '-f', 'qwen-salient-manual'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

        serve.supervise(salient_command(), [sys.executable, str(ROOT/'proxy.py'), '30001'],
                        address, common.key(), stop_container=cleanup)


def worker():
    profile = 'salient-accuracy'
    # Child logs are filtered before they are persisted, including engine startup arguments.
    secret = common.key()
    command = [sys.executable, str(ROOT/'qwen.py'), 'serve']
    log_path = STATE/'logs'/f'{profile}-{time.strftime("%Y%m%d-%H%M%S")}.log'
    process = None
    requested_stop = None

    def stop(signum, _frame):
        nonlocal requested_stop
        requested_stop = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    write_json(STATE/'active.json', {'profile': profile, 'host': common.proxy_host(), 'pid': os.getpid(),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'log': str(log_path)})
    with log_path.open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace', start_new_session=True)
        if requested_stop is not None:
            stop(requested_stop, None)
        for line in process.stdout:
            log.write(line.replace(secret, '<redacted>'))
        status = process.wait()
    write_json(STATE/'last-run.json', {'profile': profile, 'returncode': status, 'log': str(log_path)})
    return status


def start():
    profile = 'salient-accuracy'
    current = active()
    if current:
        raise RuntimeError(f'{current["profile"]} is already running; use ./qwen stop first')
    verify_preparation()
    common.key()
    process = subprocess.Popen([sys.executable, str(ROOT/'qwen.py'), 'worker'],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    for _ in range(30):
        if process.poll() is not None:
            raise RuntimeError('runner exited immediately; inspect .local/last-run.json and .local/logs')
        state = active()
        if state.get('pid') == process.pid:
            print(f'Starting {profile}; ./qwen status reports readiness. Log: {state["log"]}')
            return
        time.sleep(0.1)
    raise RuntimeError('runner did not register; inspect local logs')


def stop():
    current = active()
    if not current:
        print('No managed server is running.')
        return
    pid = current['pid']
    os.kill(pid, signal.SIGTERM)
    for _ in range(130):
        if not active():
            print('Stopped '+current['profile'])
            return
        time.sleep(0.5)
    raise RuntimeError('server did not stop within 65 seconds; inspect it before retrying')


def status():
    print('Server:', json.dumps(active() or {'state': 'stopped'}))
    try:
        models = api('/v1/models')
        print('API responding:', models['data'][0]['id'])
    except Exception:
        print('API: not ready')
    report = read_json(STATE/'preparation.json')
    print('Preparation:', json.dumps(report or {'stage': 'not started'}))


def main():
    os.umask(0o077)
    (STATE/'logs').mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    sub.add_parser('stop')
    prep = sub.add_parser('prepare')
    prep.add_argument('--background', action='store_true')
    for name in ('start', 'print', 'worker', 'serve'):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == 'status':
        return status()
    if args.command == 'print':
        command = salient_command()
        print(json.dumps({'profile': 'salient-accuracy', 'command': [str(arg) for arg in command]}, indent=2))
        return
    if args.command == 'worker':
        return worker()
    if args.command == 'serve':
        return serve_salient()
    if args.command == 'prepare':
        report = read_json(STATE/'preparation.json')
        if report.get('pid') and same_process(report['pid'], ROOT/'prepare.py'):
            raise RuntimeError('preparation is already running; ./qwen status shows progress')
        if args.background:
            with (STATE/'logs/preparation.log').open('a') as log:
                process = subprocess.Popen(['nice', '-n', '15', 'ionice', '-c', '2', '-n', '7',
                    sys.executable, str(ROOT/'prepare.py')], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (STATE/'prepare.pid').write_text(str(process.pid)+'\n')
            print('Preparation started; ./qwen status shows progress.')
            return
        os.execv(sys.executable, [sys.executable, str(ROOT/'prepare.py')])
    with (STATE/'control.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'stop':
            return stop()
        if args.command == 'start':
            return start()


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print('qwen: '+str(error), file=sys.stderr)
        sys.exit(1)
