"""Fetch a pinned upstream runtime snapshot; verify every file using git blob SHA1."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

root = Path('.tools/vendor/LIBERO-runtime')
commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
request = urllib.request.Request(
    f"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/git/trees/{commit}?recursive=1",
    headers={"User-Agent":"astra-libero-semantic-source-fetcher"})
tree = json.loads(urllib.request.urlopen(request, timeout=60).read())
if tree.get("sha") != commit or tree.get("truncated"):
    raise RuntimeError("Incomplete or unexpected upstream source tree")
selected = [v for v in tree['tree'] if v['type'] == 'blob' and (
    v['path'].startswith('libero/libero/') and not v['path'].startswith('libero/libero/init_files/')
    or v['path'] in ('setup.py', 'LICENSE', 'README.md', 'libero/__init__.py'))]
root.mkdir(parents=True, exist_ok=True)

def fetch(v):
    target = root / v['path']
    for attempt in range(3):
        try:
            if target.exists():
                data = target.read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{commit}/{v['path']}"
                data = urllib.request.urlopen(url, timeout=60).read()
            digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if digest != v['sha']:
                if target.exists(): target.unlink()
                raise ValueError('Hash mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {'path': v['path'], 'sha': v['sha'], 'ok': True}
        except Exception as e:
            error = str(e)
            time.sleep(attempt + 1)
    return {'path': v['path'], 'ok': False, 'error': error}

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for result in pool.map(fetch, selected):
        results.append(result)
        if len(results) % 50 == 0: print(f'{len(results)}/{len(selected)} verified', flush=True)
manifest = {'repository': 'Lifelong-Robot-Learning/LIBERO', 'commit': commit,
            'initialization': 'seeded procedural reset; benchmark init_files not downloaded', 'files': results}
(root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Complete:', len(results), 'Failed:', sum(not v['ok'] for v in results), flush=True)
raise SystemExit(1 if any(not v['ok'] for v in results) else 0)
