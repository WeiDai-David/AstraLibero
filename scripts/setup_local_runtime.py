"""Reproducible local Windows compatibility setup for the pinned runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import site

root = Path(__file__).resolve().parents[1]
vendor = root / '.tools/vendor/LIBERO-runtime'
if not (vendor / 'libero/libero/envs/env_wrapper.py').is_file():
    raise RuntimeError('Verified LIBERO source snapshot is missing')
Path(site.getsitepackages()[-1], 'libero_runtime.pth').write_text(str(vendor)+'\n', encoding='utf-8')
spec = importlib.util.find_spec('robosuite')
path = Path(next(iter(spec.submodule_search_locations))) / 'utils/log_utils.py'
text = path.read_text(encoding='utf-8')
logdir = root / 'outputs/runtime'
logdir.mkdir(parents=True, exist_ok=True)
old = 'logging.FileHandler("/tmp/robosuite.log")'
new = 'logging.FileHandler(' + repr(str(logdir / 'robosuite.log')) + ')'
manifest = root / '.tools/windows_compatibility.json'
if old in text:
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(text.replace(old, new), encoding='utf-8')
    manifest.write_text(json.dumps({'file':str(path), 'before_sha256':original, 'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'change':'Only relocate robosuite log file from /tmp to project outputs/runtime; controller unchanged'},indent=2),encoding='utf-8')
print('LIBERO namespace registered; robosuite log path points into project.')

base = Path(next(iter(spec.submodule_search_locations)))
patches = [
    (base / "utils/binding_utils.py", 'os.path.join(os.path.dirname(__file__), "mujoco.dll")', 'os.path.join(os.path.dirname(mujoco.__file__), "mujoco.dll")'),
    (base / "utils/binding_utils.py", 'if macros.MUJOCO_GPU_RENDERING and', 'if _SYSTEM != "Windows" and macros.MUJOCO_GPU_RENDERING and'),
]
for target, before, after in patches:
    content = target.read_text(encoding="utf-8")
    if before in content:
        target.write_text(content.replace(before, after), encoding="utf-8")
        print("Windows compatibility patch:", target.name, before, "=>", after)
