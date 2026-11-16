"""Offline checks; never prints credential values or starts paid calls."""
import argparse,importlib,shutil,sys
from config import setting

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-keys',action='store_true');parser.add_argument('--mesh',action='store_true');args=parser.parse_args()
    failed=sys.version_info<(3,12)
    print('Python:',sys.version.split()[0], '(3.12+ required)')
    for name in ['av','numpy','PIL','requests','fastapi','uvicorn','multipart','gradio_client','scipy']:
        try:importlib.import_module(name);print(name+': OK')
        except Exception as exc:print(name+': FAILED ('+type(exc).__name__+')');failed=True
    binary=shutil.which('rubberband');print('rubberband:', 'OK' if binary else 'MISSING — see docs/SETUP-AGENT.md');failed=failed or not binary
    if args.mesh:
        from pathlib import Path
        import subprocess
        worker=Path(__file__).resolve().parent/'.venv-face/bin/python'
        ready=worker.exists() and subprocess.run([str(worker),'-c','import mediapipe,av,PIL,numpy'],capture_output=True).returncode==0
        print('Face mesh worker: '+('OK' if ready else 'MISSING — run bash setup-face-mesh.sh'))
        failed=failed or not ready
    from pathlib import Path
    local_runtime=Path(__file__).resolve().parent/'.venv-models/bin/python'
    print('Local model runtime: '+('OK' if local_runtime.exists() else 'MISSING — run bash setup-local-models.sh'))
    failed=failed or not local_runtime.exists()
    sam_weights=Path(__file__).resolve().parent/'local-models/sam3/model.safetensors'
    sam_ready=sam_weights.is_file() and sam_weights.stat().st_size>=1_000_000
    print('Local SAM 3 weights: '+('OK' if sam_ready else 'MISSING — Git LFS pointer only; run git -C local-models/sam3 lfs pull'))
    failed=failed or not sam_ready
    try:
        import subprocess
        codec_ready=local_runtime.exists() and subprocess.run([str(local_runtime),'-c','import torchcodec, scipy'],capture_output=True).returncode==0
    except OSError:codec_ready=False
    print('SAM runtime extras (torchcodec, SciPy cleanup): '+('OK' if codec_ready else 'MISSING — install torchcodec and scipy'))
    failed=failed or not codec_ready
    missing=[]
    for key in ['SCNET_API_KEY']:
        value=setting(key).strip()
        valid=bool(value) and not any(x in value.lower() for x in ['your_key','your_api','replace_me','placeholder'])
        print(key+': '+('configured (not verified remotely)' if valid else 'MISSING'))
        if not valid:missing.append(key)
    if setting('MEDIA_HOST','tmpfiles') not in ['tmpfiles','catbox','scnet']:
        print('MEDIA_HOST must be tmpfiles, catbox or scnet');failed=True
    if failed:print('DEPENDENCIES NOT READY');return 1
    if missing:
        print('LOCAL UI READY; GENERATION BLOCKED until SCNET_API_KEY is configured.')
        return 2 if args.require_keys else 0
    print('LOCAL SETUP READY; provider access/balance has not been verified by this offline check.')
    return 0
if __name__=='__main__':sys.exit(main())
