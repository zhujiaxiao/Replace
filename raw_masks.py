from pathlib import Path
import json

def segment_raw(video,folder,prompt='person'):
    folder=Path(folder);out=folder/'raw-masks';out.mkdir(exist_ok=True)
    from local_models import sam3
    sam3(video,out,prompt)
    (folder/'sam-local.json').write_text(json.dumps({'provider':'local','model_path':'local-models/sam3','prompt':prompt},indent=2))
    return out

def clean_border_fragments(folder):
    """Remove tiny, narrow disconnected edge fragments; preserve large edge subjects."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label
    folder=Path(folder);output=folder.parent/'clean-masks';output.mkdir(exist_ok=True)
    report=[]
    for p in sorted(folder.glob('mask_*.png')):
        a=np.asarray(Image.open(p).convert('L'))>127;labels,_=label(a);h,w=a.shape;removed=0
        for ident in (set(labels[:,0])|set(labels[:,-1]))-{0}:
            y,x=np.where(labels==ident)
            if len(x)<h*w*.03 and x.max()-x.min()+1<=max(6,int(w*.05)):
                a[labels==ident]=False;removed+=len(x)
        Image.fromarray(a.astype(np.uint8)*255).save(output/p.name)
        report.append({'frame':p.stem,'removed_border_pixels':removed,'mask_pixels':int(a.sum()),'left_right_edge_pixels':int(a[:,0].sum()+a[:,-1].sum())})
    (folder.parent/'mask-cleanup-report.json').write_text(json.dumps(report,indent=2))
    return output

def stabilize_short_gaps(folder,max_gap=2):
    """Fill only brief empty dropouts between valid masks; never invent long tracks."""
    import numpy as np
    from PIL import Image
    folder=Path(folder);files=sorted(folder.glob('mask_*.png'))
    masks=[np.asarray(Image.open(path).convert('L'))>127 for path in files]
    valid=[bool(mask.any()) for mask in masks];filled=[];index=0
    while index<len(files):
        if valid[index]:index+=1;continue
        end=index
        while end<len(files) and not valid[end]:end+=1
        gap=end-index
        if index>0 and end<len(files) and gap<=max_gap:
            for frame in range(index,end):
                source=index-1 if frame-index<gap/2 else end
                Image.fromarray(masks[source].astype(np.uint8)*255).save(files[frame])
                masks[frame]=masks[source].copy();valid[frame]=True;filled.append(frame)
        index=end
    (folder.parent/'mask-temporal-report.json').write_text(json.dumps({'filled_frames':filled,'max_gap':max_gap,'valid_frames':sum(valid),'total_frames':len(valid)},indent=2))
    return folder
