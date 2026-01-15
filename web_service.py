import os
import uuid
import json
import base64
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from src.config.settings import ROOT_DIR
from src.stt.sensevoice_engine import SenseVoiceEngine
from src.llm.llm_engine import LLMEngine
from src.context.history import ConversationHistory
from src.context.config import HISTORY_DIR
from src.ser.schemas import EmotionLabel
from src.tts.tts_engine import TTSEngine
from src.app.pipeline import run_one_turn_from_wav_with_result, _map_sensevoice_emotion_to_label

app = FastAPI()

# ====== CORS（方便你用手机浏览器直接访问） ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 如果以后上线可改成你的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-ID"],  # 允许前端读取 X-User-ID 响应头
)

# ====== 全局引擎实例（进程级单例） ======

data_input_dir = Path(ROOT_DIR) / "data" / "web_input"
data_input_dir.mkdir(parents=True, exist_ok=True)

# TTS输出目录（按用户ID组织）
tts_output_base_dir = Path(ROOT_DIR) / "data" / "output_voice"
tts_output_base_dir.mkdir(parents=True, exist_ok=True)

try:
    stt = SenseVoiceEngine()
    llm = LLMEngine()
    tts = TTSEngine()
except RuntimeError as e:
    print("[错误] 初始化引擎失败：", e)
    raise

default_speaker_wav = os.path.join(ROOT_DIR, "src/data/input_voice/happy.wav")
speaker_wav = default_speaker_wav if Path(default_speaker_wav).exists() else None


# ====== 用户ID生成和管理 ======

def generate_user_id() -> str:
    """
    生成新的用户ID（格式：user_001, user_002, ...）
    扫描现有历史文件，找到最大编号，然后生成下一个
    """
    history_dir = HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描所有 user_*.json 文件
    max_num = 0
    for file_path in history_dir.glob("user_*.json"):
        try:
            # 从文件名提取编号：user_001.json -> 001
            filename = file_path.stem  # 去掉 .json 后缀
            num_str = filename.replace("user_", "")
            num = int(num_str)
            if num > max_num:
                max_num = num
        except (ValueError, AttributeError):
            continue
    
    # 生成下一个编号（3位数字，补零）
    next_num = max_num + 1
    return f"user_{next_num:03d}"


def get_or_create_user_id(user_id: Optional[str] = None) -> str:
    """
    获取或创建用户ID
    
    :param user_id: 如果提供，验证格式后返回；否则生成新的
    :return: 用户ID
    """
    if user_id:
        # 验证格式（可选：检查是否符合 user_XXX 格式）
        if user_id.startswith("user_") or user_id == "anonymous":
            return user_id
        else:
            # 如果格式不对，生成新的
            print(f"警告：用户ID格式不正确 '{user_id}'，将生成新的用户ID")
            return generate_user_id()
    else:
        return generate_user_id()


def get_user_input_dir(user_id: str) -> Path:
    """获取用户输入文件目录"""
    user_dir = data_input_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_tts_dir(user_id: str) -> Path:
    """获取用户TTS输出目录"""
    user_dir = tts_output_base_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def delete_user_data(user_id: str) -> dict:
    """
    删除用户的所有数据：
    - 历史记录文件
    - 用户输入目录下的所有文件
    - 用户TTS输出目录下的所有文件
    
    :param user_id: 用户ID
    :return: 删除结果统计
    """
    deleted_files = []
    deleted_dirs = []
    
    # 1. 删除历史记录文件
    history_file = HISTORY_DIR / f"{user_id}.json"
    if history_file.exists():
        try:
            history_file.unlink()
            deleted_files.append(str(history_file))
        except Exception as e:
            print(f"删除历史记录文件失败: {e}")
    
    # 2. 删除用户输入目录
    user_input_dir = data_input_dir / user_id
    if user_input_dir.exists():
        try:
            shutil.rmtree(user_input_dir)
            deleted_dirs.append(str(user_input_dir))
        except Exception as e:
            print(f"删除用户输入目录失败: {e}")
    
    # 3. 删除用户TTS输出目录
    user_tts_dir = tts_output_base_dir / user_id
    if user_tts_dir.exists():
        try:
            shutil.rmtree(user_tts_dir)
            deleted_dirs.append(str(user_tts_dir))
        except Exception as e:
            print(f"删除用户TTS输出目录失败: {e}")
    
    return {
        "user_id": user_id,
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "total_files": len(deleted_files),
        "total_dirs": len(deleted_dirs),
    }


# ====== webm/ogg -> wav 转换（用 ffmpeg） ======

def convert_to_wav(input_path: str) -> str:
    """
    不管是什么格式，都统一转成 16k 单声道 wav
    """
    output_path = str(Path(input_path).with_suffix(".wav"))
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频格式转换失败: {e}")
    return output_path


# ====== HTTP 接口：一问一答 ======

@app.get("/")
def index():
    """
    返回前端首页 index.html
    """
    index_path = Path(ROOT_DIR) / "index.html"
    if not index_path.exists():
        # 如果文件不存在，给出明确错误
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.post("/api/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    接收前端上传的一句话语音，返回 JSON 格式的响应，包含：
    - audio_base64: TTS生成的音频（base64编码）
    - user_text: 用户输入的文本
    - user_emotion: 用户情绪标签（happy/sad/angry/neutral）
    - assistant_reply: 助手回复的文本
    - assistant_emotion: 助手情绪标签
    - user_id: 用户ID
    
    请求参数：
    - file: 音频文件（multipart/form-data）
    - user_id: 用户ID（可选，如果不提供则自动生成）
    
    返回：
    - JSON 格式的响应，包含音频和识别结果
    """
    # 1. 获取或生成用户ID
    final_user_id = get_or_create_user_id(user_id)
    
    # 2. 根据用户ID创建对话历史管理器
    history = ConversationHistory(user_id=final_user_id)
    
    # 3. 获取用户输入目录
    user_input_dir = get_user_input_dir(final_user_id)
    
    suffix = Path(file.filename).suffix or ".webm"
    temp_input_path = user_input_dir / f"web_{uuid.uuid4().hex}{suffix}"

    # 保存上传文件
    try:
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"保存上传文件失败: {e}")

    wav_path = None
    tts_path = None

    try:
        # 1) 统一转成 wav
        wav_path = convert_to_wav(str(temp_input_path))

        # 2) 获取用户TTS输出目录，并生成输出路径
        user_tts_dir = get_user_tts_dir(final_user_id)
        import time
        timestamp = int(time.time())
        tts_output_path = user_tts_dir / f"tts_neutral_{timestamp}.wav"

        # 3) 走你的主流程，获取完整结果（需要修改 pipeline 函数支持指定输出路径）
        result = run_one_turn_from_wav_with_result(
            wav_path=wav_path,
            stt=stt,
            llm=llm,
            tts=tts,
            history=history,
            speaker_wav=speaker_wav,
            tts_output_path=str(tts_output_path),  # 传入指定的输出路径
        )

        # 4) 读取音频文件并转换为 base64
        with open(result["tts_path"], "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # 5) 返回 JSON 响应
        return JSONResponse({
            "user_id": final_user_id,
            "audio_base64": audio_base64,
            "user_text": result["user_text"],
            "user_emotion": result["user_emotion"],
            "assistant_reply": result["assistant_reply"],
            "assistant_emotion": result["assistant_emotion"],
        })

    except Exception as e:
        print("[错误] /api/voice-chat 处理失败：", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear-history")
async def clear_history(
    user_id: str = Form(...)
):
    """
    清除指定用户的所有数据
    
    请求参数：
    - user_id: 用户ID（必填）
    
    返回：
    - JSON 格式的响应，包含删除结果统计
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 参数不能为空")
    
    # 验证用户ID格式
    if not (user_id.startswith("user_") or user_id == "anonymous"):
        raise HTTPException(status_code=400, detail=f"无效的用户ID格式: {user_id}")
    
    try:
        # 删除用户数据
        result = delete_user_data(user_id)
        
        # 如果历史记录管理器已加载，也清空内存中的历史
        # 注意：这里只是删除文件，如果 ConversationHistory 实例已存在，需要重新加载
        
        return JSONResponse({
            "success": True,
            "message": f"已清除用户 {user_id} 的所有数据",
            "result": result,
        })
    except Exception as e:
        print(f"[错误] 清除用户 {user_id} 数据失败：", e)
        raise HTTPException(status_code=500, detail=f"清除数据失败: {str(e)}")


"""
# 安装 fastapi + uvicorn（如果还没装）
pip install fastapi "uvicorn[standard]"

# 启动
uvicorn web_service:app --host 0.0.0.0 --port 8001
python -m uvicorn web_service:app --host 0.0.0.0 --port 8001


"""