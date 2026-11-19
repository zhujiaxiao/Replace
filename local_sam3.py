from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import binary_closing,label
from PIL import Image
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
video,output,prompt=sys.argv[1],Path(sys.argv[2]),sys.argv[3];output.mkdir(parents=True,exist_ok=True)
model_dir=Path(__file__).resolve().parent/'local-models/sam3'
weights=model_dir/'model.safetensors'
if not (model_dir/'config.json').is_file():
    raise RuntimeError(f'Local SAM 3 configuration is missing: {model_dir}')
if not weights.is_file() or weights.stat().st_size<1_000_000:
    raise RuntimeError(f'Local SAM 3 weights are missing or still a Git LFS pointer: {weights}. Run git -C {model_dir} lfs pull after authenticating.')
device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
model=Sam3VideoModel.from_pretrained(model_dir,local_files_only=True).to(device).eval()
processor=Sam3VideoProcessor.from_pretrained(model_dir,local_files_only=True)
session=processor.init_video_session(video=video,inference_device=device,processing_device='cpu',video_storage_device='cpu',dtype=torch.float32)
processor.add_text_prompt(session,prompt)

def clean_union(mask):
    """Apple Silicon fallback for SAM 3 cv-utils cleanup (no macOS kernel build)."""
    mask=binary_closing(mask>0,structure=np.ones((3,3),bool))
    labels,count=label(mask,structure=np.ones((3,3),bool))
    minimum=max(16,mask.shape[0]*mask.shape[1]//10000)
    cleaned=np.zeros(mask.shape,np.uint8)
    # scipy.ndimage.label numbers components from 1 through ``count``.
    # Include the final component; when a frame contains one clean subject,
    # range(1, count) is empty and would incorrectly erase the whole mask.
    for ident in range(1,count+1):
        component=labels==ident
        if int(component.sum())>=minimum:cleaned[component]=255
    return cleaned

for raw in model.propagate_in_video_iterator(session,show_progress_bar=True):
    masks=processor.postprocess_outputs(session,raw)['masks'].detach().cpu().numpy()
    union=np.any(masks>0,axis=0).astype(np.uint8)*255 if len(masks) else np.zeros((session.video_height,session.video_width),np.uint8)
    Image.fromarray(clean_union(union)).save(output/f'mask_{raw.frame_idx:06d}.png')
