from pathlib import Path
import threading,uuid,shutil,json,time,math
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Depends,Request,Response,Cookie
from fastapi.responses import FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from pipeline import run
from task_control import JobCancelled,cancel as cancel_task,is_cancelled,register as register_task,scope as task_scope
ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
import auth
auth.init(RUNS)
app=FastAPI();jobs={}
preparation_lock=threading.Lock()
preparation_state_lock=threading.Lock();preparation_waiting=[];preparation_active=None
def workflow_mode(folder:Path,value:dict|None=None):
    """Recover the selected preparation mode, including runs created before it was persisted."""
    try:return json.loads((folder/'workflow-mode.json').read_text()).get('mode','depth')
    except (OSError,ValueError):
        status=str((value or {}).get('status',''))
        if '高级 + Face Mesh' in status or '深度 + 面部网格' in status:return 'depth_mesh'
        if '快速 Face Mesh' in status or '面部网格' in status:return 'face_mesh'
        if (folder/'depth.mp4').exists() and (folder/'face-tracking.json').exists():return 'depth_mesh'
        if (folder/'face-tracking.json').exists():return 'face_mesh'
        return (value or {}).get('mode') or 'depth'
def artifacts(folder:Path):
    """Report prepared artifacts explicitly; the UI must not depend on HEAD requests."""
    names=('upload.mp4','seedance-input.mp4','original.mp4','depth.mp4','mask.mp4','vocals.wav','vocals-plus3.wav','music.wav','audio-manifest.json','generation.json','seedance-result.mp4','final-preview.mp4','hd/final-preview.mp4')
    result={name:(folder/name).is_file() and (folder/name).stat().st_size>0 for name in names}
    for stem in ('sirio-reference','tein-reference'):
        result[stem]=next((p.name for p in folder.glob(stem+'.*') if p.is_file()),None)
    return result

def preparation_progress(message:str):
    text=message.lower()
    if 'waiting' in text:return 0
    if 'extracting original' in text:return 5
    if 'separating vocals' in text:return 15
    if 'normalizing' in text or 'preparing video' in text:return 25
    if 'tracking face mesh' in text:return 50
    if 'depth' in text and ('generating' in text or 'anything' in text):return 45
    if 'segment' in text or 'sam 3' in text:return 65
    if 'composit' in text:return 82
    if 'embedding' in text or 'pitch' in text:return 92
    if 'complete' in text or 'ready' in text:return 100
    return 1
def user_dep(user=Depends(auth.current_user)):return user
def accessible_folder(ident,user):
    if len(ident)!=32 or not all(c in '0123456789abcdef' for c in ident):raise HTTPException(404)
    folder=RUNS/ident
    if not folder.exists() or not auth.can_access(ident,user):raise HTTPException(404)
    return folder

@app.get('/')
def index(request:Request):
    try:auth.current_user(request.cookies.get('replica_session'))
    except HTTPException:
        response=RedirectResponse('/',303);set_session(response,auth.guest_user());return response
    return FileResponse(ROOT/'static/index.html')

@app.get('/login')
def login_page(request:Request):
    try:
        user=auth.current_user(request.cookies.get('replica_session'))
        if not user.get('is_guest'):return RedirectResponse('/',303)
    except HTTPException:pass
    return FileResponse(ROOT/'static/login.html')

def set_session(response:Response,user):
    response.set_cookie('replica_session',auth.create_session(user['id']),max_age=30*86400,httponly=True,samesite='lax',secure=False)
    return {'username':user['username'],'is_admin':bool(user.get('is_admin')),'is_guest':bool(user.get('is_guest'))}

@app.post('/auth/login')
def login(response:Response,username:str=Form(...),password:str=Form(...)):
    try:user=auth.authenticate(username,password)
    except ValueError as error:raise HTTPException(401,str(error))
    return set_session(response,user)

@app.post('/auth/register')
def register(response:Response,username:str=Form(...),password:str=Form(...)):
    try:user=auth.create_user(username,password)
    except ValueError as error:raise HTTPException(400,str(error))
    return set_session(response,user)

@app.post('/auth/guest')
def guest(response:Response,replica_session:str|None=Cookie(None)):
    try:
        user=auth.current_user(replica_session)
        if user.get('is_guest'):return set_session(response,user)
    except HTTPException:pass
    return set_session(response,auth.guest_user())

@app.get('/auth/me')
def me(user=Depends(auth.current_user)):return {'username':user['username'],'is_admin':bool(user['is_admin']),'is_guest':bool(user['is_guest'])}

@app.post('/auth/logout')
def logout(response:Response,replica_session:str|None=Cookie(None)):
    # Cookie is read explicitly because logout should also succeed for expired sessions.
    auth.delete_session(replica_session);response.delete_cookie('replica_session');return {'ok':True}

@app.get('/studio/status')
def studio_status(user=Depends(auth.current_user)):
    """Offline-only readiness flags. Never return credential values."""
    import shutil
    from config import setting
    model_root=ROOT/'local-models'
    missing=[]
    if not (model_root/'sam3/model.safetensors').is_file():missing.append('SAM 3')
    if not (model_root/'Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth').is_file():missing.append('Video Depth Anything')
    if not (ROOT/'.venv-models/bin/python').is_file():missing.append('本地模型环境')
    if not shutil.which('rubberband'):missing.append('Rubber Band')
    return {'key_configured':bool(setting('SCNET_API_KEY')),'local_models_ready':not missing,'missing':missing,'ui_ready':True,'provider':'SCNet'}
@app.post('/jobs')
def create(video:UploadFile=File(...),prompt:str=Form('person'),provider:str=Form('local'),mode:str=Form('depth'),user=Depends(auth.current_user)):
    if mode not in ['depth','face_mesh','depth_mesh']:raise HTTPException(400,'Unknown workflow mode')
    if provider!='local':raise HTTPException(400,'Only the local provider is supported')
    ident=uuid.uuid4().hex;folder=RUNS/ident;folder.mkdir();src=folder/'upload.mp4'
    auth.own_job(ident,user['id'])
    with src.open('wb') as out:shutil.copyfileobj(video.file,out)
    (folder/'workflow-mode.json').write_text(json.dumps({'mode':mode,'prompt':prompt}))
    jobs[ident]={'state':'running','status':'Starting local preparation','progress':0,'phase':'preparation','id':ident,'mode':mode,'source_name':video.filename or '原始视频'}
    with preparation_state_lock:preparation_waiting.append(ident)
    register_task(ident)
    (folder/'status.json').write_text(json.dumps(jobs[ident]))
    def work():
        global preparation_active
        try:
            jobs[ident].update(status='Waiting for the local model worker',progress=0)
            def update(message):
                jobs[ident].update(status=message,progress=max(jobs[ident].get('progress',0),preparation_progress(message)))
                (folder/'status.json').write_text(json.dumps(jobs[ident]))
            with task_scope(ident),preparation_lock:
                with preparation_state_lock:
                    if ident in preparation_waiting:preparation_waiting.remove(ident)
                    preparation_active=ident
                if mode=='face_mesh':
                    from face_mesh import run as run_mesh
                    run_mesh(src,folder,update)
                elif mode=='depth_mesh':
                    from face_mesh import run_depth
                    run_depth(src,folder,prompt,provider,update)
                else:run(src,folder,prompt,provider,update)
            jobs[ident].update(state='done',status='Preparation complete · input review required',progress=100)
            (folder/'status.json').write_text(json.dumps(jobs[ident]))
        except JobCancelled:
            jobs[ident].update(state='canceled',status='任务已由用户终止')
            (folder/'status.json').write_text(json.dumps(jobs[ident]))
        except Exception as e:
            canceled=is_cancelled(ident)
            jobs[ident].update(state='canceled' if canceled else 'error',status='任务已由用户终止' if canceled else str(e))
            (folder/'status.json').write_text(json.dumps(jobs[ident]))
        finally:
            with preparation_state_lock:
                if ident in preparation_waiting:preparation_waiting.remove(ident)
                if preparation_active==ident:preparation_active=None
    threading.Thread(target=work,daemon=True).start();return jobs[ident]
@app.get('/jobs/{ident}')
def get(ident:str,user=Depends(auth.current_user)):
    accessible_folder(ident,user)
    def with_provider_state(value):
        from seedance_bridge import provider_state
        value=dict(value);folder=RUNS/ident;state=provider_state(folder)
        if not value.get('mode'):
            value['mode']=workflow_mode(folder,value)
        value.setdefault('source_name','已提交原始视频')
        if value.get('state')=='running' and value.get('phase')=='draft':
            elapsed=max(0,time.time()-value.get('started_at',time.time()))
            value['progress']=max(value.get('progress',0),min(92,round(12+80*(1-math.exp(-elapsed/180)))))
        value['provider_state']=state;value['retry_allowed']=state in ['failed','cancelled','canceled'];value['cancelable']=value.get('state')=='running';value['artifacts']=artifacts(folder);return value
    if ident not in jobs:
        if len(ident)==32 and all(c in '0123456789abcdef' for c in ident) and (RUNS/ident/'status.json').exists():return with_provider_state(json.loads((RUNS/ident/'status.json').read_text()))
        raise HTTPException(404)
    return with_provider_state(jobs[ident])

@app.get('/history')
def history(user=Depends(auth.current_user)):
    """Local-only task index for the UI. Never includes credentials or provider payloads."""
    items=[];visible=auth.visible_jobs(user)
    for folder in RUNS.iterdir():
        if not folder.is_dir() or len(folder.name)!=32 or folder.name not in visible:continue
        status_file=folder/'status.json'
        if not status_file.exists():continue
        try:value=json.loads(status_file.read_text())
        except (OSError,ValueError):continue
        mode=workflow_mode(folder,value)
        try:model=json.loads((folder/'generation.json').read_text()).get('model')
        except (OSError,ValueError):model=None
        state=value.get('state','done');phase=value.get('phase','preparation')
        with preparation_state_lock:queue_state='queued' if state=='running' and phase=='preparation' and folder.name!=preparation_active else 'running' if state=='running' else 'history'
        items.append({'id':folder.name,'state':state,'queue_state':queue_state,'status':value.get('status',''),'phase':phase,'mode':mode,'model':model,'has_draft':(folder/'final-preview.mp4').is_file(),'updated_at':status_file.stat().st_mtime,'owner':auth.owner_name(folder.name) if user.get('is_admin') else None})
    items.sort(key=lambda item:item['updated_at'],reverse=True)
    return {'items':items[:50]}

@app.delete('/jobs/{ident}')
def delete_job(ident:str,user=Depends(auth.current_user)):
    folder=hd_folder(ident,user)
    current=jobs.get(ident);hd_current=hd_jobs.get(ident)
    if (current and current.get('state')=='running') or (hd_current and hd_current.get('state')=='running'):
        raise HTTPException(409,'运行中的任务请先终止。')
    shutil.rmtree(folder)
    jobs.pop(ident,None);hd_jobs.pop(ident,None);auth.forget_job(ident)
    return {'id':ident,'deleted':True}
@app.post('/jobs/{ident}/seedance')
def seedance(ident:str,prompt:str=Form(...),model:str=Form('Minimax-H3-LumaCraft'),retry_failed:bool=Form(False),sirio:UploadFile=File(...),tein:UploadFile|None=File(None),user=Depends(auth.current_user)):
    folder=accessible_folder(ident,user)
    if not (folder/'audio-manifest.json').exists():raise HTTPException(400,'Audio separation must complete first.')
    from mask_review import require_approved
    try:require_approved(folder)
    except ValueError as e:raise HTTPException(409,str(e))
    paths=[]
    for name,upload in [('sirio-reference',sirio),('tein-reference',tein)]:
        if upload is None:
            paths.append(None)
            continue
        suffix=Path(upload.filename or '').suffix.lower()
        if suffix not in (['.png','.jpg','.jpeg','.webp','.mp4','.mov','.webm'] if name=='sirio-reference' else ['.png','.jpg','.jpeg','.webp']):raise HTTPException(400,'Use an image or a Sirio reference video.')
        path=folder/(name+suffix)
        with path.open('wb') as f:shutil.copyfileobj(upload.file,f)
        paths.append(path)
    from seedance_bridge import submit
    try:request_id=submit(folder,prompt,*paths,model=model,retry_failed=retry_failed)
    except Exception as e:raise HTTPException(502,str(e))
    jobs[ident]={'id':ident,'state':'running','status':'SCNet 正在生成草稿 · 进度为估算值','progress':12,'phase':'draft','started_at':time.time()}
    register_task(ident)
    def collect():
        from seedance_bridge import wait_and_finish
        try:
            with task_scope(ident):wait_and_finish(folder,request_id,lambda s:jobs[ident].update(status=s,progress=96))
            jobs[ident].update(state='done',status='Final preview ready · entire original source audio',progress=100,final_url='/runs/'+ident+'/final-preview.mp4')
        except JobCancelled:jobs[ident].update(state='canceled',status='已停止跟踪该生成任务；服务商任务可能仍在运行')
        except Exception as e:
            canceled=is_cancelled(ident)
            jobs[ident].update(state='canceled' if canceled else 'error',status='已停止跟踪该生成任务；服务商任务可能仍在运行' if canceled else str(e))
        (folder/'status.json').write_text(json.dumps(jobs[ident]))
    threading.Thread(target=collect,daemon=True).start()
    return {'request_id':request_id,'id':ident,'state':'running','status':jobs[ident]['status']}

@app.post('/jobs/{ident}/cancel')
def cancel(ident:str,user=Depends(auth.current_user)):
    folder=hd_folder(ident,user)
    hd_target=hd_jobs.get(ident)
    target=hd_target if hd_target and hd_target.get('state')=='running' else jobs.get(ident)
    if target is None and (folder/'status.json').exists():target=json.loads((folder/'status.json').read_text())
    if not target or target.get('state')!='running':raise HTTPException(409,'当前任务不在运行，无需终止。')
    cancel_task(ident)
    provider_submitted=bool((folder/'seedance-response.json').exists() or (folder/'hd/response.json').exists())
    message='已终止任务；服务商任务可能仍在运行，已产生的费用不会撤销。' if provider_submitted else '任务已由用户终止'
    target.update(state='canceled',status=message,cancelable=False)
    (folder/'cancelled.json').write_text(json.dumps({'at':time.time(),'provider_submitted':provider_submitted,'status':message}))
    status_file=folder/'hd/status.json' if target is hd_target else folder/'status.json'
    status_file.parent.mkdir(exist_ok=True);status_file.write_text(json.dumps(target))
    return {'id':ident,'state':'canceled','status':message,'provider_submitted':provider_submitted}

@app.post('/jobs/{ident}/prompt-preview')
def prompt_preview(ident:str,prompt:str=Form(...),model:str=Form('Minimax-H3-LumaCraft'),reference_is_video:bool=Form(False),user=Depends(auth.current_user)):
    folder=hd_folder(ident,user)
    from seedance_bridge import VIDEO_MODELS,build_prompt
    if model not in VIDEO_MODELS:raise HTTPException(400,'Unsupported SCNet video model.')
    _,full_prompt=build_prompt(folder,prompt,reference_is_video)
    from seedance_bridge import MINIMAX_MODELS
    return {'prompt':full_prompt,'model':model,'resolution':'768P' if model in MINIMAX_MODELS else '720p','reference_is_video':reference_is_video,'provider':'SCNet'}

@app.post('/jobs/{ident}/finish')
def finish(ident:str,video:UploadFile=File(...),user=Depends(auth.current_user)):
    folder=accessible_folder(ident,user)
    if not (folder/'upload.mp4').exists():raise HTTPException(400,'Original source is required.')
    target=folder/'seedance-result.mp4'
    with target.open('wb') as f:shutil.copyfileobj(video.file,f)
    from audio_workflow import finish_preview
    try:finish_preview(target,folder)
    except Exception as e:raise HTTPException(400,str(e))
    return {'url':'/runs/'+ident+'/final-preview.mp4'}

@app.get('/runs/{ident}/{asset_path:path}')
def run_asset(ident:str,asset_path:str,user=Depends(auth.current_user)):
    folder=accessible_folder(ident,user).resolve();target=(folder/asset_path).resolve()
    if folder not in target.parents or not target.is_file():raise HTTPException(404)
    return FileResponse(target)

hd_jobs={}
hd_lock=threading.Lock()

def hd_folder(ident,user=None):
    if len(ident)!=32 or not all(c in '0123456789abcdef' for c in ident):raise HTTPException(404)
    folder=RUNS/ident
    if not folder.exists():raise HTTPException(404)
    if user is not None and not auth.can_access(ident,user):raise HTTPException(404)
    return folder

@app.get('/jobs/{ident}/hd')
def get_hd(ident:str,user=Depends(auth.current_user)):
    folder=hd_folder(ident,user)
    if ident in hd_jobs:return hd_jobs[ident]
    saved=folder/'hd/status.json'
    if saved.exists():return json.loads(saved.read_text())
    return {'state':'idle','status':'Approve your draft to generate the model-specific high-resolution export.'}

@app.post('/jobs/{ident}/hd')
def create_hd(ident:str,user=Depends(auth.current_user)):
    folder=hd_folder(ident,user)
    try:model=json.loads((folder/'seedance-request.json').read_text()).get('model','')
    except (OSError,ValueError):model=''
    from seedance_bridge import MINIMAX_MODELS
    target='2K' if model in MINIMAX_MODELS else '1080p'
    with hd_lock:
        previous=get_hd(ident,user)
        if previous['state'] in ['running','done']:return previous
        if not (folder/'final-preview.mp4').exists():raise HTTPException(400,'Complete and review a draft first.')
        hd=folder/'hd';hd.mkdir(exist_ok=True)
        hd_jobs[ident]={'state':'running','status':f'Submitting approved draft for {target}','id':ident,'resolution':target}
        register_task(ident)
        def save(): (hd/'status.json').write_text(json.dumps(hd_jobs[ident]))
        save()
    def work_hd():
        from seedance_bridge import submit_hd,wait_and_finish
        try:
            with task_scope(ident):
                request_id=submit_hd(folder)
                hd_jobs[ident].update(request_id=request_id,status=f'Generating {target} through SCNet');save()
                wait_and_finish(hd,request_id,lambda s:hd_jobs[ident].update(status=s))
            hd_jobs[ident].update(state='done',status=f'{target} ready · original source audio restored')
        except JobCancelled:hd_jobs[ident].update(state='canceled',status='已停止跟踪该高清任务；服务商任务可能仍在运行')
        except Exception as e:
            canceled=is_cancelled(ident)
            hd_jobs[ident].update(state='canceled' if canceled else 'error',status='已停止跟踪该高清任务；服务商任务可能仍在运行' if canceled else str(e))
        save()
    threading.Thread(target=work_hd,daemon=True).start()
    return hd_jobs[ident]

@app.get('/jobs/{ident}/mask-review')
def get_mask_review(ident:str,user=Depends(auth.current_user)):
    from mask_review import state
    return state(hd_folder(ident,user))

@app.post('/jobs/{ident}/mask-review')
def save_mask_review(ident:str,approved:bool=Form(...),fingerprint:str=Form(...),note:str=Form(''),user=Depends(auth.current_user)):
    from mask_review import review
    try:return review(hd_folder(ident,user),approved,note,fingerprint)
    except ValueError as e:raise HTTPException(409,str(e))

app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

@app.on_event('startup')
def resume_saved_generations():
    """Resume collection only: never requeue a paid prediction on restart."""
    from seedance_bridge import provider_state,saved_task_id,wait_and_finish
    for root in RUNS.iterdir():
        if not root.is_dir() or len(root.name)!=32:continue
        if (root/'cancelled.json').exists():continue
        for folder,response_file,registry in [(root,'seedance-response.json',jobs),(root/'hd','response.json',hd_jobs)]:
            response=folder/response_file
            if not response.exists() or (folder/'final-preview.mp4').exists():continue
            state=provider_state(folder)
            if state in ['failed','cancelled','canceled']:
                prepared=folder/'seedance-input.mp4';provider_status=folder/'provider-status.json'
                if prepared.exists() and provider_status.exists() and prepared.stat().st_mtime>provider_status.stat().st_mtime:
                    registry[root.name]={'id':root.name,'state':'done','status':'Prepared input changed after the failed SCNet attempt · full-clip review required before retry'}
                else:registry[root.name]={'id':root.name,'state':'error','status':'Saved SCNet task ended with provider status '+state}
                continue
            request_id=saved_task_id(folder,response_file)
            if not request_id:continue
            registry[root.name]={'id':root.name,'state':'running','status':'正在继续获取已提交的 SCNet 草稿','progress':35,'phase':'draft','started_at':time.time()}
            register_task(root.name)
            def collect_saved(folder=folder,ident=root.name,request_id=request_id,registry=registry):
                try:
                    with task_scope(ident):wait_and_finish(folder,request_id,lambda message:registry[ident].update(status=message))
                    registry[ident].update(state='done',status='Final preview ready · original source audio')
                except JobCancelled:registry[ident].update(state='canceled',status='已停止跟踪该生成任务；服务商任务可能仍在运行')
                except Exception as e:
                    canceled=is_cancelled(ident)
                    registry[ident].update(state='canceled' if canceled else 'error',status='已停止跟踪该生成任务；服务商任务可能仍在运行' if canceled else str(e))
                (folder/'status.json').write_text(json.dumps(registry[ident]))
            threading.Thread(target=collect_saved,daemon=True).start()

@app.get('/health')
def health():
    return {'status':'ok','service':'复刻镜'}
