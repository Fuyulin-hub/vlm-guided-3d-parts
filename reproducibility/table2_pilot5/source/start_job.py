"""Start exactly one detached worker on a sufficiently free GPU, never killing other jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
if (root / 'worker.pid').exists():
    raise RuntimeError('Worker PID already recorded; inspect it before attempting a restart')
lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
free = [tuple(map(int, line.split(','))) for line in lines]
gpu, available = max(free, key=lambda pair: pair[1])
if available < 22000:
    raise RuntimeError('No GPU currently has the required 22,000 MiB available; no job launched')
env = dict(os.environ, PYTHONUNBUFFERED='1')
with (root / 'driver.log').open('a') as log:
    process = subprocess.Popen([sys.executable, str(root / 'source/run_batch.py'), '--root', str(root), '--gpu', str(gpu)],
        cwd=str(root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
(root / 'worker.pid').write_text(str(process.pid))
print(json.dumps({'worker_pid': process.pid, 'physical_gpu': gpu, 'free_MiB_at_launch': available, 'root': str(root)}))
