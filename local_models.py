from pathlib import Path
import shutil, subprocess, sys, tempfile
from task_control import run as controlled_run
ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-models/bin/python'
VDA = ROOT / 'local-models/Video-Depth-Anything'
SAM3 = ROOT / 'local-models/sam3'

def _runtime():
    if not PYTHON.exists(): raise RuntimeError('Local models are not installed. Run: bash setup-local-models.sh')

def demucs(audio, output_dir):
    _runtime(); output_dir=Path(output_dir); work=output_dir/'demucs-output'
    controlled_run([str(PYTHON),'-m','demucs','--two-stems=vocals','-n','htdemucs','--float32','-d','mps' if sys.platform=='darwin' else 'cpu','-o',str(work),str(audio)])
    stems=work/'htdemucs'/Path(audio).stem
    shutil.copy2(stems/'vocals.wav',output_dir/'vocals.wav');shutil.copy2(stems/'no_vocals.wav',output_dir/'music.wav')

def depth(video,destination):
    _runtime(); checkpoint=VDA/'checkpoints/video_depth_anything_vits.pth'
    if not checkpoint.exists(): raise RuntimeError('Video Depth Anything checkpoint missing; rerun setup-local-models.sh')
    out=Path(destination).parent/'local-depth-output';out.mkdir(exist_ok=True)
    controlled_run([str(PYTHON),str(VDA/'run.py'),'--input_video',str(Path(video).resolve()),'--output_dir',str(out.resolve()),'--encoder','vits','--target_fps','24','--max_res','960','--max_len','-1','--fp32'],cwd=VDA)
    result=out/(Path(video).stem+'_vis.mp4')
    if not result.exists(): raise RuntimeError('Local depth model produced no visualization video')
    shutil.copy2(result,destination)

def sam3(video,output_dir,prompt):
    _runtime()
    weights=SAM3/'model.safetensors'
    if not weights.is_file() or weights.stat().st_size<1_000_000:
        raise RuntimeError(f'Local SAM 3 weights are unavailable or still a Git LFS pointer: {weights}')
    env=__import__('os').environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    controlled_run([str(PYTHON),str(ROOT/'local_sam3.py'),str(video),str(output_dir),prompt],env=env)

def sam3_detects(video,prompt,probe_frames=8):
    """Cheap local probe to avoid running SAM 3 over an unusable depth clip."""
    _runtime()
    with tempfile.TemporaryDirectory(prefix='sam3-probe-') as temporary:
        temporary=Path(temporary);sample=temporary/'sample.mp4';masks=temporary/'masks'
        controlled_run(['ffmpeg','-v','error','-y','-i',str(video),'-frames:v',str(probe_frames),'-an',str(sample)])
        sam3(sample,masks,prompt)
        from PIL import Image
        import numpy as np
        return any(np.asarray(Image.open(path).convert('L')).any() for path in masks.glob('mask_*.png'))
