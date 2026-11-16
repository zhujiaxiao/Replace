"""Optional local face-mesh preparation; no depth or segmentation calls."""
from pathlib import Path
import json,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parent

def run(src,folder,status=print):
    from pipeline import normalize
    from audio_workflow import separate,prepare_seedance_video
    folder=Path(folder)
    python=ROOT/'.venv-face/bin/python'
    if not python.exists():raise ValueError('Fast face mesh needs setup: bash setup-face-mesh.sh (Python 3.12).')
    folder.mkdir(parents=True,exist_ok=True)
    status('Normalizing video');normalize(src,folder/'original.mp4')
    status('Tracking face mesh locally · no depth or SAM')
    try:subprocess.run([str(python),str(ROOT/'mesh_worker.py'),str(folder)],check=True)
    except subprocess.CalledProcessError as exc:
        report=folder/'face-tracking.json';data=json.loads(report.read_text()) if report.exists() else {}
        tracked=data.get('tracked',0);frames=data.get('frames',0)
        raise ValueError(f'快速 Face Mesh 跟踪失败：只在 {tracked}/{frames} 帧检测到脸部。该模式仅适合单个、居中、脸部持续清晰可见的说话者；大幅运动、转身、遮挡或脸部较小的视频请改用高级深度 + SAM。') from exc
    (folder/'workflow-mode.json').write_text(json.dumps({'mode':'face_mesh'}))
    separate(src,folder,status)
    prepare_seedance_video(folder)
    status('Face mesh ready · review tracking and pitched audio')


def run_depth(src,folder,prompt='person',provider='local',status=print):
    from pipeline import run as run_advanced
    from audio_workflow import prepare_seedance_video
    folder=Path(folder)
    python=ROOT/'.venv-face/bin/python'
    if not python.exists():raise ValueError('Install face mesh first: bash setup-face-mesh.sh')
    folder.mkdir(parents=True,exist_ok=True)
    run_advanced(src,folder,prompt,provider,status)
    shutil.copy2(folder/'composite-silent.mp4',folder/'depth-composite-silent.mp4')
    status('Adding face mesh from original video onto depth composite')
    try:subprocess.run([str(python),str(ROOT/'mesh_worker.py'),str(folder),'--depth'],check=True)
    except subprocess.CalledProcessError as exc:
        report=folder/'face-tracking.json';data=json.loads(report.read_text()) if report.exists() else {}
        raise ValueError(f'高级 + Face Mesh 跟踪失败：只在 {data.get("tracked",0)}/{data.get("frames",0)} 帧检测到脸部。请使用脸部持续清晰可见的视频，或改用不含 Face Mesh 的高级深度模式。') from exc
    (folder/'workflow-mode.json').write_text(json.dumps({'mode':'depth_mesh'}))
    prepare_seedance_video(folder)
    status('Depth + face mesh ready · full mask and facial tracking review required')
