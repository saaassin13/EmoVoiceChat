import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path（用于直接运行此文件时）
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional
from src.config.settings import ROOT_DIR
from src.audio.record import Recorder
from src.audio.player import play_wav
from src.stt.sensevoice_engine import SenseVoiceEngine
from src.llm.llm_engine import LLMEngine
from src.context.history import ConversationHistory
from src.ser.schemas import EmotionLabel
from src.tts.tts_engine import TTSEngine
from src.tts.config import get_emotion_params

# 兼容原来的简单 demo：录音 + Whisper 转写
from src.stt.whisper_engine import WhisperEngine


def record_and_transcribe() -> None:
    """原有 demo：录音 + Whisper 转写（保留）"""
    data_dir = Path("data/input")
    data_dir.mkdir(parents=True, exist_ok=True)
    wav_path = data_dir / "latest.wav"

    # 1. 录音
    rec = Recorder()
    out_path = rec.record(str(wav_path), max_duration=30.0)
    rec.close()

    # 2. Whisper 转写
    stt = WhisperEngine()
    result = stt.transcribe(out_path)

    print("\n========== 转写结果 ==========")
    print(result.text)
    print("================================")


def _map_sensevoice_emotion_to_label(emotion: Optional[str]) -> EmotionLabel:
    """
    将 SenseVoice 返回的情绪标签（HAPPY/SAD/ANGRY/NEUTRAL 或 None）
    映射到项目内的 EmotionLabel（happy/sad/angry/neutral）
    """
    if not emotion:
        return EmotionLabel.NEUTRAL
    e = emotion.upper()
    mapping = {
        "HAPPY": EmotionLabel.HAPPY,
        "SAD": EmotionLabel.SAD,
        "ANGRY": EmotionLabel.ANGRY,
        "NEUTRAL": EmotionLabel.NEUTRAL,
        "CALM": EmotionLabel.NEUTRAL,
    }
    return mapping.get(e, EmotionLabel.NEUTRAL)


def run_one_turn(
    recorder: Recorder,
    stt: SenseVoiceEngine,
    llm: LLMEngine,
    tts: TTSEngine,
    history: ConversationHistory,
    speaker_wav: Optional[str] = None,
) -> Optional[str]:
    """
    单轮：录音 -> STT+情绪 -> LLM 回复 -> 记录历史 -> TTS 合成
    返回生成的语音文件路径（供播放）
    """
    data_dir = Path("data/input")
    data_dir.mkdir(parents=True, exist_ok=True)
    wav_path = data_dir / "latest.wav"

    # 1. 录音（静音自动停止）
    out_path = recorder.record(str(wav_path), max_duration=30.0)

    # todo 测试使用
    # out_path = "/home/yanlan/workspace/ai/index-tts/data/voice_clone1_calm.wav"

    # 2. SenseVoice 转写 + 情绪
    sv_result = stt.transcribe(str(out_path))
    user_text = sv_result.text
    user_emotion = _map_sensevoice_emotion_to_label(sv_result.emotion)

    print("\n========== STT+情绪 ==========")
    print("文本:", user_text)
    print("情绪标签(SenseVoice):", sv_result.emotion)
    print("映射后 EmotionLabel:", user_emotion.value)
    print("================================")

    if not user_text.strip():
        user_text = "用户什么都没有说"
        print("没有识别到有效文本")
        # return None

    # 3. 取最近对话历史
    recent_turns = history.get_recent_turns()

    # 4. LLM 生成回复
    try:
        reply = llm.generate_reply(
            user_text=user_text,
            user_emotion=user_emotion,
            history=recent_turns,
        )
    except RuntimeError as e:
        # 粗暴但明确的 OOM 报错提示，不做自动降级（避免猜测）
        if "out of memory" in str(e).lower():
            print("\n[错误] LLM 推理发生显存 OOM：")
            print("  - 请尝试：")
            print("    1) 换更小的模型（例如改成 1.8B/3B 级别）")
            print("    2) 使用 llama.cpp 后端 + gguf 量化模型")
            print("    3) 改为 CPU 推理（会很慢，但更稳）")
        else:
            print("\n[错误] LLM 推理失败：", e)
        raise SystemExit(1)
    
    try:
        reply_dict = json.loads(reply)
    except Exception as e:
        reply_dict = {"replay": "", "emotion": ""}
        print(f"llm 回答格式错误: {reply}")

    print("\n========== LLM 回复 ==========")
    print(reply_dict)
    assistant_reply = reply_dict["replay"]
    assistant_emotion = reply_dict["emotion"]
    assistant_emotion = _map_sensevoice_emotion_to_label(assistant_emotion)
    print("================================")

    # 5. 记录历史
    history.add_turn(
        user_text=user_text,
        user_emotion=user_emotion,
        assistant_reply=assistant_reply,
    )

    tts_path = tts.synthesize(
        text=assistant_reply,
        emotion=assistant_emotion,
        speaker_wav=speaker_wav,
        # speed=params.speed,
        # pitch / volume 目前各后端支持情况不一致，这里先不乱传（避免猜测）
    )

    print("\n========== TTS 结果 ==========")
    print("情绪:", assistant_emotion.value)
    print("语音文件:", tts_path)
    print("================================")

    # 使用 pyaudio 播放合成好的语音
    try:
        play_wav(tts_path)
    except Exception as e:
        print(f"[警告] 播放语音失败：{e}")

    return tts_path


def run_chat_loop() -> None:
    """
    交互式语音聊天循环：
    每次按回车开始说话，自动录音 -> STT+情绪 -> LLM -> TTS。
    """
    recorder = Recorder()
    stt = SenseVoiceEngine()
    history = ConversationHistory(user_id="anonymous")
    tts = TTSEngine()  # 默认 indextts 后端，对应 HTTP 服务
    try:
        llm = LLMEngine()
    except RuntimeError as e:
        print("[错误] 初始化 LLMEngine 失败：", e)
        print("请检查：模型是否太大 / 显存是否足够 / 后端配置是否正确。")
        recorder.close()
        tts.unload()
        return

    default_speaker_wav = os.path.join(ROOT_DIR, "src/data/input_voice/happy.wav")
    speaker_wav = default_speaker_wav if Path(default_speaker_wav).exists() else None
    print(f"speaker_wav : {speaker_wav}")

    print("\n=== 语音聊天已启动 ===")
    print("回车开始录音，输入 q 后回车退出。")

    try:
        while True:
            cmd = input("\n按回车开始新一轮对话（q + 回车退出）：").strip().lower()
            if cmd == "q":
                break

            run_one_turn(
                recorder=recorder,
                stt=stt,
                llm=llm,
                tts=tts,
                history=history,
                speaker_wav=speaker_wav,
            )
    finally:
        recorder.close()
        tts.unload()
        llm.unload()
        print("已退出语音聊天。")


def run_one_turn_from_wav_with_result(
    wav_path: str,
    stt: SenseVoiceEngine,
    llm: LLMEngine,
    tts: TTSEngine,
    history: ConversationHistory,
    speaker_wav: Optional[str] = None,
    tts_output_path: Optional[str] = None,  # 新增：指定TTS输出路径
) -> dict:
    """
    不负责录音，仅做：
      STT+情绪 -> LLM -> 历史记录 -> TTS
    返回包含完整结果的字典：
    {
        "tts_path": str,  # TTS生成的语音文件路径
        "user_text": str,  # 用户输入的文本
        "user_emotion": str,  # 用户情绪标签（happy/sad/angry/neutral）
        "assistant_reply": str,  # 助手回复的文本
        "assistant_emotion": str,  # 助手情绪标签
    }
    """
    sv_result = stt.transcribe(wav_path)
    user_text = sv_result.text
    user_emotion = _map_sensevoice_emotion_to_label(sv_result.emotion)

    print("\n========== STT+情绪 ==========")
    print("文本:", user_text)
    print("情绪标签(SenseVoice):", sv_result.emotion)
    print("映射后 EmotionLabel:", user_emotion.value)
    print("================================")

    if not user_text.strip():
        user_text = "用户什么都没有说"
        print("没有识别到有效文本")

    recent_turns = history.get_recent_turns()

    try:
        reply = llm.generate_reply(
            user_text=user_text,
            user_emotion=user_emotion,
            history=recent_turns,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n[错误] LLM 推理发生显存 OOM：", e)
        else:
            print("\n[错误] LLM 推理失败：", e)
        raise

    try:
        reply_dict = json.loads(reply)
    except Exception:
        reply_dict = {"replay": "", "emotion": ""}
        print(f"llm 回答格式错误: {reply}")

    assistant_reply = reply_dict.get("replay", "")
    assistant_emotion_raw = reply_dict.get("emotion", "")
    assistant_emotion = _map_sensevoice_emotion_to_label(assistant_emotion_raw)

    print("\n========== LLM 回复 ==========")
    print(assistant_reply)
    print("情绪:", assistant_emotion.value)
    print("================================")

    history.add_turn(
        user_text=user_text,
        user_emotion=user_emotion,
        assistant_reply=assistant_reply,
    )

    # 如果指定了输出路径，使用指定的路径；否则使用默认路径
    tts_path = tts.synthesize(
        text=assistant_reply,
        emotion=assistant_emotion,
        speaker_wav=speaker_wav,
        output_path=tts_output_path,  # 传入指定的输出路径
    )

    print("\n========== TTS 结果 ==========")
    print("语音文件:", tts_path)
    print("================================")

    return {
        "tts_path": tts_path,
        "user_text": user_text,
        "user_emotion": user_emotion.value,
        "assistant_reply": assistant_reply,
        "assistant_emotion": assistant_emotion.value,
    }


if __name__ == "__main__":
    # 默认跑完整语音聊天流程
    run_chat_loop()
    # 如需仅测试录音+Whisper，可单独调用：
    # record_and_transcribe()