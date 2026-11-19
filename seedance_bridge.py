"""SCNet asynchronous video-generation adapter."""
from pathlib import Path
import concurrent.futures,datetime,json,shutil,time
import av,requests
from config import credential,setting
from media_host import upload
from audio_workflow import finish_preview
from task_control import checkpoint

VIDEO_MODELS=('Minimax-H3-LumaCraft','Minimax-H3-Max-LumaCraft','Seedance2.0','SeeDance-2-5-LumaCraft','SeeDance-2.0-LumaCraft')
MINIMAX_MODELS=('Minimax-H3-LumaCraft','Minimax-H3-Max-LumaCraft')
MODEL_CATALOG=VIDEO_MODELS+('Qwen-Image-3.0-LumaCraft','Qwen3-Embedding-8B')
PROMPT_SUFFIX='\nThe reference video is the prepared source with isolated vocals pitched +3 semitones; preserve its performance, dialogue timing, camera, crop and background. '
LEGACY_PROMPT_SUFFIX='\nThe first reference video is the prepared source with isolated vocals pitched +3 semitones; preserve its performance, dialogue timing, camera, crop and background. '

def _base():return setting('SCNET_BASE_URL','https://api.scnet.cn/api/llm/v1').rstrip('/')
def _headers():return {'Authorization':'Bearer '+credential('SCNET_API_KEY'),'Content-Type':'application/json','X-MultiModal-Async':'true'}

TERMINAL_FAILURES={'failed','cancelled','canceled'}

def build_prompt(folder,prompt,identity_is_video=False):
    """Build the exact provider prompt without uploading or submitting media."""
    from mask_review import is_mesh
    user_prompt=prompt.split(PROMPT_SUFFIX,1)[0].split(LEGACY_PROMPT_SUFFIX,1)[0].rstrip()
    identity=('The second reference video supplies visual identity only; do not use its audio or performance.' if identity_is_video else 'The reference image supplies replacement appearance only. Do not copy its pose, gesture, framing, background, text, logo or watermark; take the opening pose and all motion only from the prepared source video.')
    full_prompt=user_prompt+PROMPT_SUFFIX+identity
    full_prompt+=' Purple, pink and false-color depth or mask regions are control guidance only. They must disappear completely in every frame and must never be visible in the output. Replace the complete original identity, including hair, hands, clothing and footwear, across cuts and fast motion.'
    if is_mesh(folder):full_prompt+=' Remove all face mesh lines; use them only as motion guidance.'
    return user_prompt,full_prompt

def saved_task_id(folder,response_name='seedance-response.json'):
    """Return a saved SCNet task id, while retaining legacy Enhancor compatibility."""
    response=Path(folder)/response_name
    if not response.exists():return None
    try:data=json.loads(response.read_text())
    except (ValueError,OSError):return None
    return str(data.get('requestId') or (data.get('output') or {}).get('task_id') or '') or None

def provider_state(folder):
    status_file=Path(folder)/'provider-status.json'
    if not status_file.exists():return None
    try:data=json.loads(status_file.read_text())
    except (ValueError,OSError):return None
    return str((data.get('output') or {}).get('task_status') or data.get('status') or '').lower() or None

def _archive_failed_attempt(folder):
    folder=Path(folder);state=provider_state(folder)
    if state not in TERMINAL_FAILURES:
        raise ValueError('Only a definitively failed or cancelled SCNet task can start a new paid attempt.')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    attempt=folder/'attempts'/('attempt-'+stamp)
    suffix=1
    while attempt.exists():
        suffix+=1;attempt=folder/'attempts'/(f'attempt-{stamp}-{suffix}')
    attempt.mkdir(parents=True)
    for name in ['seedance-request.json','seedance-response.json','provider-status.json','generation.json','submission-error.json']:
        source=folder/name
        if source.exists():shutil.move(str(source),attempt/name)
    return attempt

def _video_parameters(path,resolution='720p'):
    with av.open(str(path)) as source:
        seconds=float(source.duration/av.time_base);stream=source.streams.video[0];ratio=stream.width/stream.height
    ratios={'16:9':16/9,'9:16':9/16,'1:1':1,'4:3':4/3,'3:4':3/4,'21:9':21/9}
    # SCNet rejects durations below 4 and above 15. Short prepared clips use
    # the minimum 4-second generation; finish_preview later removes only the
    # provider-added ending frames while restoring the complete source audio.
    duration=max(4,min(15,round(seconds)))
    parameters={'generate_audio':True,'ratio':min(ratios,key=lambda r:abs(ratios[r]-ratio)),'duration':duration,'watermark':False,'camera_fixed':False}
    if resolution:parameters['resolution']=resolution
    return parameters

def _queue(folder,payload,response_name='seedance-response.json'):
    response=requests.post(_base()+'/videos/generations',headers=_headers(),json=payload,timeout=90)
    if not response.ok:
        (Path(folder)/'submission-error.json').write_text(response.text)
        raise ValueError(f'SCNet rejected submission ({response.status_code}): {response.text[:500]}')
    data=response.json();(Path(folder)/response_name).write_text(json.dumps(data,indent=2))
    task_id=(data.get('output') or {}).get('task_id')
    if not task_id:raise ValueError('SCNet did not return output.task_id.')
    return str(task_id)

def submit(folder,prompt,sirio,tein=None,status=print,model=None,retry_failed=False):
    folder=Path(folder)
    from mask_review import require_approved,is_mesh
    require_approved(folder)
    if not (folder/'audio-manifest.json').exists():raise ValueError('Audio separation must complete before submission.')
    model=model or setting('SCNET_VIDEO_MODEL','Minimax-H3-LumaCraft')
    if model not in VIDEO_MODELS:raise ValueError('The selected model is not supported by the SCNet video endpoint.')
    if (folder/'seedance-response.json').exists():
        if not retry_failed:raise ValueError('This job already has a provider response. Resume it; use the explicit failed-task retry only for a new paid attempt.')
        archived=_archive_failed_attempt(folder)
        status('Archived failed SCNet attempt in '+str(archived.relative_to(folder)))
    from audio_workflow import prepare_seedance_video
    status('Pitching isolated vocals +3 semitones and embedding in the prepared video')
    prepared=folder/'seedance-input.mp4' if is_mesh(folder) else prepare_seedance_video(folder)
    status('Uploading prepared video and character references')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:urls=list(pool.map(upload,[prepared,sirio]+([tein] if tein else [])))
    identity_is_video=Path(sirio).suffix.lower() in ['.mp4','.mov','.webm']
    if identity_is_video:
        media=[{'type':'reference_video','url':urls[0]},{'type':'reference_video','url':urls[1]}]
    else:
        # Match SCNet's working multimodal example: identity image first, motion video second.
        media=[{'type':'reference_image','url':urls[1]},{'type':'reference_video','url':urls[0]}]
    if tein:media.append({'type':'reference_image','url':urls[2]})
    # Older UI versions reloaded the provider-expanded prompt into the editable field.
    # Strip that adapter-owned suffix before rebuilding it so retries remain idempotent.
    user_prompt,full_prompt=build_prompt(folder,prompt,identity_is_video)
    # SCNet requires Minimax H3 to use its own case-sensitive resolution enum.
    # Keep Seedance's established draft setting unchanged.
    resolution='768P' if model in MINIMAX_MODELS else '720p'
    payload={'model':model,'input':{'prompt':full_prompt,'media':media},'parameters':_video_parameters(prepared,resolution)}
    (folder/'seedance-request.json').write_text(json.dumps(payload,indent=2))
    task_id=_queue(folder,payload)
    (folder/'generation.json').write_text(json.dumps({'request_id':task_id,'prompt':full_prompt,'user_prompt':user_prompt,'video':'seedance-result.mp4','mode':'multi_reference' if identity_is_video else 'reference_video','provider':'SCNet','model':model},indent=2))
    status('SCNet video task submitted: '+task_id);return task_id

def wait_and_finish(folder,request_id,status=print):
    folder=Path(folder)
    for _ in range(360):
        checkpoint()
        response=requests.get(_base()+'/tasks/'+str(request_id),headers=_headers(),timeout=30);response.raise_for_status();data=response.json()
        (folder/'provider-status.json').write_text(json.dumps(data,indent=2));output=data.get('output') or {};state=str(output.get('task_status','')).lower()
        if state=='succeeded':
            results=output.get('results') or [];url=results[0] if results else None
            if not isinstance(url,str) or not url.startswith('http'):raise ValueError('SCNet result URL missing.')
            media=requests.get(url,timeout=180);media.raise_for_status();(folder/'seedance-result.mp4').write_bytes(media.content)
            status('Restoring entire original source audio · discarding generated audio');finish_preview(folder/'seedance-result.mp4',folder);return
        if state in ['failed','cancelled','canceled']:raise RuntimeError('SCNet video failed: '+str(output.get('error_message') or output.get('error_code') or state))
        for _ in range(20):time.sleep(.5);checkpoint()
    raise TimeoutError('SCNet task is still pending. Task ID saved; do not resubmit automatically.')

def submit_hd(folder):
    """Explicitly approved high-resolution rerender; SCNet has no draft-upgrade endpoint."""
    folder=Path(folder);hd=folder/'hd'
    if (hd/'response.json').exists():raise ValueError('High-resolution export already submitted. Resume its saved task.')
    if not (folder/'final-preview.mp4').exists():raise ValueError('A completed preview is required before generating 1080p.')
    payload=json.loads((folder/'seedance-request.json').read_text())
    payload['parameters']['resolution']='2K' if payload.get('model') in MINIMAX_MODELS else '1080p'
    hd.mkdir(exist_ok=True);__import__('shutil').copy2(folder/'upload.mp4',hd/'upload.mp4');(hd/'request.json').write_text(json.dumps(payload,indent=2))
    return _queue(hd,payload,'response.json')
