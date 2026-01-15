"""麦克风录音，保存 wav（16kHz 单通道）"""
import time
import wave
from typing import Optional, Callable, List

import numpy as np
import pyaudio

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit


class Recorder:
    """
    简单麦克风录音器：
    - 16kHz、单通道、16-bit
    - 支持最大录音时长
    - 支持“静音自动停止”（基于简单能量阈值）
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = 1024,
        silence_threshold: float = 500.0,
        silence_duration: float = 4.0,
        device_index: Optional[int] = None,
    ) -> None:
        """
        :param sample_rate: 采样率（默认 16k）
        :param channels: 通道数（默认单声道）
        :param chunk_size: 每次从缓冲区读取的帧数
        :param silence_threshold: 静音判定阈值（越大越不敏感）
        :param silence_duration: 连续静音多久（秒）后自动停止
        :param device_index: 录音设备索引，None 表示使用默认设备
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.device_index = device_index

        self._pa = pyaudio.PyAudio()

    def list_input_devices(self) -> List[str]:
        """列出所有输入设备，方便用户选择"""
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append(f"{i}: {info['name']}")
        return devices

    def _is_silence(self, data: bytes) -> bool:
        """基于简单能量判定是否静音"""
        audio_np = np.frombuffer(data, dtype=np.int16)
        if audio_np.size == 0:
            return True
        energy = np.abs(audio_np).mean()
        return energy < self.silence_threshold

    def record(
        self,
        output_path: str,
        max_duration: float = 30.0,
        on_chunk: Optional[Callable[[bytes], None]] = None,
        print_progress: bool = True,
    ) -> str:
        """
        开始录音并保存为 wav 文件。

        :param output_path: 输出 wav 文件路径
        :param max_duration: 最大录音时长（秒）
        :param on_chunk: 每次读取一块音频时回调（可选，用于实时可视化等）
        :param print_progress: 是否打印简单进度
        :return: 最终保存的 wav 文件路径
        """
        stream = self._pa.open(
            format=self._pa.get_format_from_width(DEFAULT_SAMPLE_WIDTH),
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index,
        )

        frames: List[bytes] = []
        start_time = time.time()
        last_non_silence_time = start_time

        if print_progress:
            print(f"开始录音，按 Ctrl+C 强制停止，或静音 {self.silence_duration}s 自动停止...")

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

                now = time.time()

                # 进度输出
                if print_progress:
                    elapsed = now - start_time
                    print(f"\r录音中：{elapsed:5.1f}s", end="", flush=True)

                # 回调处理（可选）
                if on_chunk is not None:
                    on_chunk(data)

                # 静音检测
                if not self._is_silence(data):
                    last_non_silence_time = now
                else:
                    if now - last_non_silence_time >= self.silence_duration:
                        if print_progress:
                            print(f"\n检测到静音超过 {self.silence_duration}s，自动停止录音。")
                        break

                # 最大时长限制
                if now - start_time >= max_duration:
                    if print_progress:
                        print(f"\n达到最大录音时长 {max_duration}s，自动停止录音。")
                    break

        except KeyboardInterrupt:
            if print_progress:
                print("\n用户手动停止录音。")
        finally:
            stream.stop_stream()
            stream.close()

        # 保存 wav
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._pa.get_sample_size(self._pa.get_format_from_width(DEFAULT_SAMPLE_WIDTH)))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        if print_progress:
            print(f"录音已保存到：{output_path}")

        return output_path

    def close(self) -> None:
        """释放资源"""
        self._pa.terminate()


if __name__ == "__main__":
    rec = Recorder()
    print("可用输入设备：")
    for dev in rec.list_input_devices():
        print("  ", dev)

    # 简单测试：录一段，保存到当前目录
    rec.record("test_record.wav", max_duration=30.0)
    rec.close()