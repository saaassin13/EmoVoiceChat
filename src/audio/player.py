"""扬声器播放合成好的 wav"""

import wave
from typing import Optional

import pyaudio


def play_wav(path: str, device_index: Optional[int] = None) -> None:
    """
    使用 pyaudio 播放 wav 文件（阻塞至播放结束）

    :param path: wav 文件路径
    :param device_index: 输出设备索引，None 表示默认设备
    """
    wf = wave.open(path, "rb")

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=device_index,
    )

    chunk = 1024
    data = wf.readframes(chunk)

    try:
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        wf.close()