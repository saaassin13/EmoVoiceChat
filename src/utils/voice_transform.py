from pydub import AudioSegment
import os

def convert_audio(input_path, output_path, bitrate="192k"):
    """
    音频格式转换函数
    :param input_path: 输入音频文件路径（如：./input.wav）
    :param output_path: 输出音频文件路径（如：./output.mp3）
    :param bitrate: 输出码率（有损格式有效，如128k/192k/320k，越高音质越好但文件越大）
    """
    try:
        # 1. 读取输入音频（pydub会自动识别输入格式）
        audio = AudioSegment.from_file(input_path)
        
        # 2. 导出为目标格式（根据输出后缀自动选择编码）
        # 关键参数说明：
        # format：可选值"wav"/"mp3"/"m4a"/"flac"/"ogg"等
        # bitrate：仅对有损格式（mp3/m4a）有效
        # codec：指定编码器，m4a建议用"aac"
        output_format = os.path.splitext(output_path)[1].lstrip('.')
        export_kwargs = {"format": output_format, "bitrate": bitrate}
        if output_format == "m4a":
            export_kwargs["codec"] = "aac"
        
        audio.export(output_path, **export_kwargs)
        
        print(f"转换成功！输出文件：{output_path}")
        return True
    except Exception as e:
        print(f"转换失败：{str(e)}")
        return False
        

# 示例调用
if __name__ == "__main__":
    # 示例1：WAV转MP3（320k高码率）
    # convert_audio("input.wav", "output.mp3", bitrate="320k")
    
    # # 示例2：MP3转M4A
    # convert_audio("input.mp3", "output.m4a")
    
    # # 示例3：FLAC（无损）转WAV
    # convert_audio("input.flac", "output.wav")

    convert_audio("src/data/input_voice/angry.m4a", "src/data/input_voice/angry.wav")