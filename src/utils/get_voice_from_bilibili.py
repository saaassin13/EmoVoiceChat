"""
给定链接地址，从Bilibili获取音频文件
然后根据时间段截取音频文件

https://github.com/nilaoda/BBDown?tab=readme-ov-file
# 下载对应架构的二进制文件
wget https://github.com/nilaoda/BBDown/releases/download/1.6.3/BBDown_1.6.3_20240814_linux-x64.zip
unzip BBDown_1.6.3_20240814_linux-x64.zip
chmod +x BBDown
sudo mv BBDown /usr/local/bin/

BBDown --audio-only "BV号"
"""

from pydub import AudioSegment

def trim_audio(input_file, output_file, start_ms, end_ms):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    
    # 截取指定时间段
    trimmed_audio = audio[start_ms:end_ms]
    
    # 保存截断后的音频
    trimmed_audio.export(output_file, format="wav")
    print(f"音频已保存到 {output_file}")

if __name__ == "__main__":
    # # 使用示例（时间单位为毫秒）
    # input_file = "/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/qinche_long.wav"
    # output_file = "/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/qinche.wav"
    # trim_audio(input_file, output_file, 30000, 45000)  # 截取30-45秒

    # 使用示例（时间单位为毫秒）
    input_file = "/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/mabaoguo_long.wav"
    output_file = "/home/yanlan/workspace/code/emo-voice-chat/src/data/input_voice/mabaoguo.wav"
    trim_audio(input_file, output_file, 37000, 47000) 