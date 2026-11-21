"""Foreground engine/proxy supervisor with bounded shutdown and no restart policy."""
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.request




class Stopped(Exception):
    def __init__(self, signum):
        self.signum = signum


def stopped(signum, _frame):
    raise Stopped(signum)


def child_failed(children):
    for label, process in children:
        status = process.poll()
        if status is not None:
            raise RuntimeError(f'{label} exited with status {status}')


def ready(url, secret, models=False):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + secret})
        with opener.open(request, timeout=2) as response:
            body = response.read()
            return response.status == 200 and (not models or any(
                row.get('id') == 'qwen3.8-flash-next' for row in json.loads(body).get('data', [])))
    except Exception:
        return False


def supervise(engine, proxy, address, secret, stop_container=None, timeout=1800):
    children = []
    try:
        for label, command in [('engine', engine), ('proxy', proxy)]:
            children.append((label, subprocess.Popen(command, start_new_session=True)))
        connect_host = '127.0.0.1' if address == '0.0.0.0' else address
        deadline, report = time.monotonic() + timeout, 0
        while True:
            child_failed(children)
            if ready('http://127.0.0.1:30000/health', secret) and ready(
                f'http://{connect_host}:30001/v1/models', secret, models=True):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'engine/proxy not ready after {timeout}s')
            if time.monotonic() >= report:
                print('Waiting for engine readiness.', flush=True)
                report = time.monotonic() + 30
            time.sleep(1)
        print(f'READY: http://{address}:30001\nModel: qwen3.8-flash-next\nCtrl+C stops the engine and proxy.', flush=True)
        while True:
            child_failed(children)
            time.sleep(0.5)
    finally:
        # Further signals must not interrupt cleanup halfway through.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if stop_container:
            try:
                stop_container()
            except Exception as exc:
                print(f'Container cleanup failed: {exc}', file=sys.stderr)
        for _, process in children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while any(p.poll() is None for _, p in children) and time.monotonic() < deadline:
            time.sleep(0.1)
        for _, process in children:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
