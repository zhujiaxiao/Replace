# 复刻镜 AI · 视频角色替换

复刻镜 AI 是支持桌面和手机浏览器的视频角色替换工作台，整合本地输入准备、全片审核、角色参考、SCNet 云端草稿生成及高清导出。

## 主要功能

- 深度 + SAM、面部网格、深度 + 面部网格三种准备方式。
- 本地运行 Video Depth Anything、SAM 3、Demucs 和 MediaPipe。
- 分离人声并升调三个半音，保持时间；结果恢复完整原始音轨。
- 原片、合成输入、深度、Mask、草稿预览与对比，问题帧标记。
- 任务队列、终止、历史删除及显式批准的独立高清生成。
- 注册登录、自动游客模式、任务和媒体访问隔离、管理员查看全部任务。

本地准备无需 Replicate API；视频生成依赖 SCNet API，提交草稿和高清任务可能计费。

## 技术栈与依赖

前端：HTML、CSS、原生 JavaScript ES Modules、HTML5 Video、SVG。
后端：Python、FastAPI、Uvicorn、SQLite。
处理：PyTorch、Transformers、PyAV、FFmpeg、Rubber Band、NumPy、Pillow、OpenCV、SciPy。

推荐 Python 3.12，系统还需 Git、FFmpeg/FFprobe、Rubber Band CLI。cloudflared 用于当前启动器的兼容回调隧道。

全部 Python 依赖统一维护在 `requirement.txt`，主应用和两个独立工作环境均从此文件安装。模型权重和系统命令仍需单独安装。

默认模型为 `Minimax-H3-LumaCraft`，其次是 `Minimax-H3-Max-LumaCraft`，再列出 Seedance 系列。历史任务仍沿用其保存的模型。

## 安装与启动

### 1. 获取仓库并安装系统依赖

克隆自己的 GitLab 仓库并进入目录。macOS 可执行：

```bash
brew install python@3.12 ffmpeg rubberband cloudflared git-lfs
```

Ubuntu/Debian 请安装 Python 3.12、venv、FFmpeg、Rubber Band，并按对应系统安装 cloudflared。

### 2. 主应用环境

```bash
PYTHON=python3.12 bash setup.sh
```

也可以手动安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
cp .env.example .env
```

### 3. 本地模型环境

```bash
PYTHON=python3.12 bash setup-local-models.sh
FACE_PYTHON=python3.12 bash setup-face-mesh.sh
```

脚本下载 Video Depth Anything 仓库和小型权重。SAM 3 需自行获取有权访问的完整模型目录，放置为：

```text
local-models/sam3/model.safetensors
local-models/sam3/...模型配置与处理器文件
local-models/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth
```

SAM 3 必须包含完整权重和配置，不能仅有 Git LFS 指针。Demucs 和 Face Landmarker 首次使用可能下载模型。预留足够内存、推理资源和磁盘空间。

### 4. 配置服务

编辑项目的 .env：

```dotenv
SCNET_API_KEY=填写自己的密钥
SCNET_BASE_URL=https://api.scnet.cn/api/llm/v1
SCNET_VIDEO_MODEL=Minimax-H3-LumaCraft
MEDIA_HOST=scnet
MEDIA_HOST_FALLBACKS=
```

模型以账号实际权限和界面列表为准。Seedance 使用 720p 草稿、1080p 高清；Minimax H3 使用 768P 草稿、2K 高清。

### 5. 检查与运行

```bash
.venv/bin/python doctor.py
bash start.command
```

也可运行：

```bash
.venv/bin/python launch.py
```

浏览器打开 http://127.0.0.1:8770/ 。更换端口：

```bash
PORT=8771 bash start.command
```

网页启动成功不代表本地模型和生成权限已就绪。安装阶段不要提交付费任务验证环境。

## 使用步骤

1. 打开网页，默认进入游客模式；右上角可登录或注册。
2. 选择原始视频，选择准备方式；深度模式填写替换主体描述。
3. 点击“开始映射”，等待本地准备。手机选择视频后进入素材准备区。
4. 完整检查合成输入、Mask/面部跟踪和音频，确认后批准输入。问题帧标记仅用于定位。
5. 上传角色参考，编辑复刻描述，选择模型。
6. 创建草稿并核对完整提示词，确认后提交；手机自动切换镜面预览查看进度。
7. 完成后验收和下载，结果自动恢复原始音轨。
8. 如需高清，另行批准独立生成任务。

面部网格适合单个居中、持续露脸的人物。转身、遮挡和小脸可能导致跟踪失败；重做输入后需重新审核。

任务队列显示运行、排队和历史任务。终止会停止本地处理或轮询，服务商任务仍可能继续计费。删除会清理对应本机文件。

## 目录说明

- app.py、auth.py：Web API、用户和权限。
- pipeline.py、local_models.py、local_sam3.py：深度和主体分割。
- face_mesh.py、mesh_worker.py：面部网格。
- audio_workflow.py：人声处理及原始音轨恢复。
- seedance_bridge.py：SCNet 请求与结果收集。
- task_control.py：任务取消。
- static/：前端；tests/：测试；docs/：技术说明，部分为历史上游文档。
- local-models/、models/：模型；runs/、app.db：私有运行数据，均忽略提交。

## 许可

保留 LICENSE、THIRD-PARTY.md 及上游声明。模型、第三方库和生成服务受各自许可及服务条款约束。
