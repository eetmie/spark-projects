"""Supply private state to the upstream proxy, in this process only."""
import os
import runpy
from common import CONFIG, STATE, VENDOR, key

os.environ['QWEN38_UPSTREAM_API_KEY'] = key()
os.environ['QWEN38_API_KEY_FILE'] = str(CONFIG / 'api-key')
# Disable upstream service/pair discovery for this single, manually managed engine.
os.environ['PROXY_HOLD_UNITS'] = ''
os.environ['QWEN38_PAIR_FILE'] = str(STATE / 'pair-disabled.json')
runpy.run_path(str(VENDOR / 'keepalive-proxy.py'), run_name='__main__')
