"""
https://github.com/index-tts/index-tts/blob/main/docs/README_zh.md
tts服务使用的是bilibili开源的index-tts模型，需要下载模型文件到本地，然后运行tts_service.py文件，即可启动tts服务。
将此文件cp到tts_index根目录下，然后运行tts_service.py文件，即可启动tts服务。

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import time
from indextts.infer_v2 import IndexTTS2

app = FastAPI(title="IndexTTS2 API", version="1.0.0")

# 全局 TTS 实例（启动时初始化）
tts: Optional[IndexTTS2] = None

class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    spk_audio_prompt: str = Field(..., description="音色参考音频文件路径")
    output_path: Optional[str] = Field(None, description="输出音频文件路径，如果不提供则自动生成")
    
    # 可选参数
    emo_audio_prompt: Optional[str] = Field(None, description="情感参考音频文件路径")
    emo_alpha: float = Field(1.0, description="情感权重", ge=0.0, le=1.0)
    emo_vector: Optional[List[float]] = Field(None, description="情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    use_emo_text: bool = Field(False, description="是否使用情感描述文本控制")
    emo_text: Optional[str] = Field(None, description="情感描述文本")
    use_random: bool = Field(False, description="是否使用情感随机采样")
    interval_silence: int = Field(200, description="段落间静音时长（毫秒）")
    verbose: bool = Field(False, description="是否输出详细信息")
    max_text_tokens_per_segment: int = Field(120, description="每个分句的最大token数")
    
    # Generation kwargs
    do_sample: Optional[bool] = Field(None, description="是否进行采样")
    top_p: Optional[float] = Field(None, description="Top-p采样参数", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Top-k采样参数", ge=0)
    temperature: Optional[float] = Field(None, description="温度参数", ge=0.1, le=2.0)
    length_penalty: Optional[float] = Field(None, description="长度惩罚")
    num_beams: Optional[int] = Field(None, description="Beam search的beam数量", ge=1)
    repetition_penalty: Optional[float] = Field(None, description="重复惩罚", ge=0.1)
    max_mel_tokens: Optional[int] = Field(None, description="最大mel token数", ge=50)

class TTSResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化 TTS 模型"""
    global tts
    try:
        cfg_path = "checkpoints/config.yaml"
        model_dir = "/home/yanlan/.cache/modelscope/hub/models/IndexTeam/IndexTTS-2"
        
        # 检查路径是否存在
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False
        )
        print(">> TTS 模型初始化成功")
    except Exception as e:
        print(f">> TTS 模型初始化失败: {e}")
        raise

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "tts_loaded": tts is not None
    }

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    文本转语音接口
    
    - **text**: 要合成的文本
    - **spk_audio_prompt**: 音色参考音频文件路径
    - **output_path**: 输出音频文件路径（可选，不提供则自动生成）
    """
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS 模型未初始化")
    
    # 检查输入音频文件是否存在
    if not os.path.exists(request.spk_audio_prompt):
        raise HTTPException(
            status_code=400,
            detail=f"音色参考音频文件不存在: {request.spk_audio_prompt}"
        )
    
    # 如果提供了情感参考音频，检查是否存在
    if request.emo_audio_prompt and not os.path.exists(request.emo_audio_prompt):
        raise HTTPException(
            status_code=400,
            detail=f"情感参考音频文件不存在: {request.emo_audio_prompt}"
        )
    
    # 如果没有提供输出路径，自动生成
    if not request.output_path:
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        request.output_path = os.path.join(output_dir, f"tts_output_{timestamp}.wav")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(request.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 构建 generation_kwargs
        generation_kwargs = {}
        if request.do_sample is not None:
            generation_kwargs["do_sample"] = request.do_sample
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        if request.length_penalty is not None:
            generation_kwargs["length_penalty"] = request.length_penalty
        if request.num_beams is not None:
            generation_kwargs["num_beams"] = request.num_beams
        if request.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = request.repetition_penalty
        if request.max_mel_tokens is not None:
            generation_kwargs["max_mel_tokens"] = request.max_mel_tokens
        
        # 调用 TTS 推理
        result = tts.infer(
            spk_audio_prompt=request.spk_audio_prompt,
            text=request.text,
            output_path=request.output_path,
            emo_audio_prompt=request.emo_audio_prompt,
            emo_alpha=request.emo_alpha,
            emo_vector=request.emo_vector,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text,
            use_random=request.use_random,
            interval_silence=request.interval_silence,
            verbose=request.verbose,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            **generation_kwargs
        )
        
        # 检查输出文件是否生成成功
        if result and os.path.exists(request.output_path):
            return TTSResponse(
                success=True,
                message="语音合成成功",
                output_path=request.output_path
            )
        else:
            return TTSResponse(
                success=False,
                message="语音合成失败，未生成输出文件",
                output_path=None
            )
            
    except Exception as e:
        error_msg = f"语音合成过程中发生错误: {str(e)}"
        print(f">> {error_msg}")
        return TTSResponse(
            success=False,
            message=error_msg,
            output_path=None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)