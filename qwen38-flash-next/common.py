"""Pinned, standalone Flash-Next recipe. Python standard library only on the host."""
import ipaddress
import os
from pathlib import Path
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'
NAME = 'qwen-salient-manual'
MODEL_NAME = 'qwen3.8-flash-next'
STATE = Path(os.environ.get('QWEN_FLASH_STATE', ROOT / '.local')).expanduser().resolve()
CONFIG = STATE / 'config'


def call(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def output(args):
    return call(args, capture_output=True, text=True, timeout=20).stdout.strip()


def proxy_host():
    return str(ipaddress.IPv4Address(os.environ.get('QWEN_HOST', '127.0.0.1')))


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('cannot read MemAvailable')


def check_port_available(host, port):
    with socket.socket() as sock:
        # Match the real HTTP servers: recently closed connections in TIME_WAIT
        # must not prevent restarting inference. A live listener still fails.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as error:
            raise RuntimeError(f'serving port unavailable: {host}:{port}: {error}') from error


def preflight():
    for binary in ('docker', 'nvidia-smi', 'bash', 'timeout'):
        if not shutil.which(binary):
            raise RuntimeError(f'{binary} is required')
    gpu = output(['nvidia-smi', '--query-gpu=name,driver_version,addressing_mode', '--format=csv,noheader'])
    if len(gpu.splitlines()) != 1 or 'GB10' not in gpu:
        raise RuntimeError(f'this recipe requires a single GB10 GPU; found {gpu}')
    fields = [s.strip() for s in gpu.split(',')]
    if fields[-1] != 'ATS':
        raise RuntimeError(f'file-backed PLE requires ATS; found {fields[-1]}')
    if int(fields[1].split('.')[0]) < 580:
        raise RuntimeError('the CUDA 13 image requires NVIDIA driver branch 580 or newer')
    call(['bash', VENDOR / 'engine-preflight.sh', NAME])
    apps = output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    # Allow the remote-desktop daemon when it uses at most 1 GiB.
    busy = []
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(',')]
        desktop = (len(parts) == 3 and parts[1] == '/usr/libexec/gnome-remote-desktop-daemon'
                   and parts[2].isdigit() and int(parts[2]) <= 1024)
        if not desktop:
            busy.append(line)
    if busy:
        raise RuntimeError('GPU compute processes are already running:\n' + '\n'.join(busy))
    memory = available_gib()
    if memory < 96:
        raise RuntimeError(f'need at least 96 GiB available before startup; found {memory:.1f}')
    address = proxy_host()
    for host, port in [('127.0.0.1', 30000), (address, 30001)]:
        check_port_available(host, port)
    output(['docker', 'info', '--format', '{{.ServerVersion}}'])
    existing = output(['docker', 'ps', '-aq', '--filter', f'name=^/{NAME}$'])
    if existing:
        raise RuntimeError(f'container {NAME} already exists; inspect it before running again')
    mount = output(['findmnt', '-n', '-o', 'SOURCE', '-T', STATE])
    if not mount.startswith('/dev/nvme'):
        raise RuntimeError(f'state must be on local NVMe for file-backed PLE; found {mount}')
    free = shutil.disk_usage(STATE).free / 2**30
    print(f'Preflight: {gpu}; {memory:.1f} GiB available; {free:.1f} GiB disk free; API bind {address}', flush=True)
    return address


def key():
    path = CONFIG / 'api-key'
    if path.stat().st_mode & 0o077:
        raise RuntimeError(f'{path} must have mode 600')
    value = path.read_text().strip()
    if not value or any(ord(c) < 33 or ord(c) > 126 or c in '\"\\' for c in value):
        raise RuntimeError(f'invalid key in {path}; run ./qwen prepare')
    return value
