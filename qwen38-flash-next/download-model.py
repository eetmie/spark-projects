"""CPU-only, resumable download and full SHA256 verification of the pinned model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.request

MODEL = 'nvidia/Qwen3.8-Flash-Next-NVFP4'
REVISION = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'


def main():
    spec = importlib.util.spec_from_file_location('fetch', '/helpers/ckpt-fetch.py')
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    snapshot = None
    for attempt in range(1, 13):
        try:
            # The upstream helper retries with Xet for the checkpoint's >50 GB shard.
            with fetch.reporting(MODEL, revision=REVISION):
                snapshot = Path(fetch.download(MODEL, REVISION, {'max_workers': 2}))
            if not fetch.complete(snapshot):
                raise RuntimeError('checkpoint indexes name missing weight files')
            break
        except Exception as error:
            print(f'Download attempt {attempt}/12 interrupted: {type(error).__name__}', flush=True)
            if attempt == 12:
                raise
            time.sleep(30)
    request = urllib.request.Request(
        f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true',
        headers=({'Authorization': 'Bearer '+os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}))
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise RuntimeError('Hub returned an unexpected revision')
    print('Download complete. Verifying all LFS files by SHA256.', flush=True)
    checked = []
    for item in metadata['siblings']:
        lfs = item.get('lfs') or {}
        expected = lfs.get('sha256') or lfs.get('oid')
        if not expected:
            continue
        path = snapshot/item['rfilename']
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                hasher.update(chunk)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if hasher.hexdigest() != expected:
            raise RuntimeError('SHA256 mismatch: '+item['rfilename'])
        checked.append({'file': item['rfilename'], 'sha256': expected, 'bytes': path.stat().st_size})
        print('Verified '+item['rfilename'], flush=True)
    if not checked:
        raise RuntimeError('Hub metadata supplied no LFS hashes to verify')
    manifest = {'model': MODEL, 'revision': REVISION, 'files': checked,
                'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    Path('/hf/verified-model.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Pinned checkpoint downloaded and verified.', flush=True)


if __name__ == '__main__':
    main()
