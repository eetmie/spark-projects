"""Download only; no GPU access, model startup, service installation or switching."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('QWEN_FLASH_STATE', ROOT/'.local')).expanduser().resolve()
RECIPE = ROOT/'recipe'
IMAGE = 'ghcr.io/salient-data/qwen38-next-dgx@sha256:d55a2740651064f6c8aba20bc9ada104ae9df0a793cc87d0c170d312aa052b4d'
REVISION = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'
COSIGN_VERSION = 'v3.1.4'
COSIGN_HASH = 'ade83b7aac62e131cacfe96d958948b2150cc9efec46a570a2649b8e27dedad2'
import common


class AlreadyRunning(Exception):
    pass


def record(**values):
    path = STATE/'preparation.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current, indent=2)+'\n')
    temporary.replace(path)


def call(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    os.umask(0o077)
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE/'prepare.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunning('preparation is already running') from None
        record(stage='image-signature', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), passed=False, error=None)
        hf = STATE/'huggingface'
        hf.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(STATE).free < 180 * 2**30 and not (hf/'verified-model.json').exists():
            raise RuntimeError('need at least 180 GiB free for a fresh preparation')
        binary = STATE/'bin/cosign'
        binary.parent.mkdir(exist_ok=True)
        if not binary.exists() or hashlib.sha256(binary.read_bytes()).hexdigest() != COSIGN_HASH:
            print('Downloading checksum-pinned Cosign '+COSIGN_VERSION, flush=True)
            temporary = binary.with_suffix('.download')
            with urllib.request.urlopen(f'https://github.com/sigstore/cosign/releases/download/{COSIGN_VERSION}/cosign-linux-arm64', timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != COSIGN_HASH:
                raise RuntimeError('Cosign binary checksum mismatch')
            temporary.chmod(0o700)
            temporary.replace(binary)
        print('Verifying the published image signature before pulling.', flush=True)
        for attempt in range(1, 4):
            try:
                call([binary, '--timeout', '5m', 'verify', '--key', RECIPE/'cosign.pub', IMAGE],
                     timeout=330, stdout=subprocess.DEVNULL)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt == 3:
                    raise
                print('Signature verification interrupted; retrying without relaxing verification.', flush=True)
                time.sleep(30)
        record(stage='image-download', signature_verified=True)
        for attempt in range(1, 7):
            try:
                call(['docker', 'pull', IMAGE])
                break
            except subprocess.CalledProcessError:
                if attempt == 6:
                    raise
                time.sleep(30)
        identity = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE], text=True).strip()
        if identity not in ('sha256:3eda66c8163064d664465a803ca09fb66ab626814b6bba7e58a6fd97a3cde282', IMAGE.split('@')[1]):
            raise RuntimeError('published image identity does not match the audited pins')
        common.CONFIG.mkdir(parents=True, exist_ok=True, mode=0o700)
        keyfile = common.CONFIG/'api-key'
        if not keyfile.exists():
            with keyfile.open('x') as output:
                output.write(secrets.token_urlsafe(32)+'\n')
        keyfile.chmod(0o600)
        common.key()
        record(stage='model-download')
        print('Downloading NVIDIA checkpoint; CPU only, two workers, 3 GiB memory cap.', flush=True)
        cidfile = STATE/'download.cid'
        cidfile.unlink(missing_ok=True)
        command = ['docker', 'run', '--rm', '--init', '--cidfile', str(cidfile), '--name', 'qwen-model-download',
                   '--network', 'host', '--user', f'{os.getuid()}:{os.getgid()}',
                   '--memory', '3g', '--memory-swap', '3g', '--cpus', '2', '--pids-limit', '256',
                   '--entrypoint', 'python3', '-e', 'HF_HOME=/hf', '-e', 'HF_HUB_DISABLE_XET=1',
                   '-e', 'HF_HUB_DOWNLOAD_TIMEOUT=60', '-e', 'HF_XET_NUM_CONCURRENT_RANGE_GETS=4',
                   '-e', 'HF_XET_HIGH_PERFORMANCE=0', '-e', 'HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1',
                   '-e', 'RAYON_NUM_THREADS=2', '-v', f'{hf}:/hf',
                   '-v', f'{ROOT/"vendor"}:/helpers:ro', '-v', f'{ROOT/"download-model.py"}:/download-model.py:ro']
        if os.environ.get('HF_TOKEN'):
            command += ['-e', 'HF_TOKEN']
        command += [IMAGE, '/download-model.py']
        try:
            call(command)
        finally:
            if cidfile.exists():
                cid = cidfile.read_text().strip()
                if cid:
                    subprocess.run(['docker', 'rm', '-f', cid], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=30)
                cidfile.unlink(missing_ok=True)
        snapshot = hf/'hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots'/REVISION
        call(['sha256sum', '-c', RECIPE/'model-files.sha256'], cwd=snapshot)
        record(stage='prepared', passed=True, model_verified=True, image=IMAGE, image_identity=identity,
               gpu_runtime_tested=False, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        print('PREPARED: model hashes, image signature and image identity verified. No model was started.', flush=True)


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    except AlreadyRunning as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
    except BaseException as error:
        record(stage='failed', passed=False, error=f'{type(error).__name__}: {error}')
        print('Preparation failed; downloads can resume: '+str(error), file=sys.stderr, flush=True)
        sys.exit(1)
