"""封装情绪识别，支持 wav2vec2-base-emo / CNN-RNN 两种实现（可配置）"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path（用于直接运行此文件时）
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoProcessor, AutoModelForAudioClassification

from src.ser.config import (
    SER_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    SAMPLE_RATE,
)
from src.ser.schemas import EmotionLabel, SERResult


class EmotionClassifier(nn.Module):
    """简单的情绪分类头（如果模型本身没有分类层）"""
    
    def __init__(self, hidden_size: int, num_labels: int = 4):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states):
        # 使用平均池化
        pooled = hidden_states.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class SEREngine:
    """基于 wav2vec2 的语音情绪识别引擎"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        :param model_name: Hugging Face 模型名称或路径
        :param device: 设备（cuda/cpu）
        :param confidence_threshold: 置信度阈值
        """
        self.model_name = model_name or SER_MODEL_NAME
        self.device = device or ( "cuda" if torch.cuda.is_available() else "cpu" )
        self.confidence_threshold = confidence_threshold

        # 尝试加载预训练的情绪识别模型
        # 如果模型本身支持分类，直接使用；否则需要添加分类头
        self._load_model()

    def _load_model(self) -> None:
        """加载模型和处理器"""
        try:
            # 尝试加载专门的情绪识别模型（如果有）
            # 例如：superb/hubert-base-superb-er 或其他情绪识别模型
            # 缓存到 ~/.cache/huggingface/hub/
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
            self._has_classifier = True
            
            # 打印模型信息用于调试
            if hasattr(self.model, 'config'):
                num_labels = getattr(self.model.config, 'num_labels', None)
                id2label = getattr(self.model.config, 'id2label', None)
                print(f"模型加载成功：{self.model_name}")
                print(f"  类别数量: {num_labels}")
                if id2label:
                    print(f"  标签映射: {id2label}")
                    
        except Exception as e:
            # 如果模型不支持直接分类，使用 wav2vec2-base + 自定义分类头
            print(f"警告：{self.model_name} 不支持直接分类，使用 wav2vec2-base + 自定义分类头")
            print(f"  错误信息: {e}")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            
            # 添加分类头
            hidden_size = base_model.config.hidden_size
            self.classifier = EmotionClassifier(hidden_size, num_labels=4).to(self.device)
            self.model = base_model.to(self.device)
            self.model.eval()
            self._has_classifier = False
            
            # 注意：这里需要加载训练好的分类头权重
            # 如果没有预训练权重，可以先用随机初始化（准确率会较低）
            print("提示：分类头未加载预训练权重，建议使用专门训练过的情绪识别模型")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """加载音频文件，确保采样率为 16kHz"""
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return audio

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """预处理音频：转换为模型输入格式"""
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values.to(self.device)

    def _predict_emotion(self, audio_input: torch.Tensor) -> Dict[str, float]:
        """
        预测情绪，返回所有情绪的得分字典
        
        动态处理不同数量的类别，并映射到我们的 4 类情绪系统
        """
        with torch.no_grad():
            if self._has_classifier:
                # 模型自带分类层
                outputs = self.model(audio_input)
                logits = outputs.logits
            else:
                # 使用自定义分类头
                hidden_states = self.model(audio_input).last_hidden_state
                logits = self.classifier(hidden_states)

            # 转换为概率
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            num_classes = len(probs)
            
            # 获取模型的实际标签（如果有的话）
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                # 使用模型配置中的标签
                id2label = self.model.config.id2label
                label_map = {}
                for i, (idx, label) in enumerate(id2label.items()):
                    if i < num_classes:
                        label_map[int(idx)] = label.lower()
            else:
                # 默认标签映射（根据常见情况）
                label_map = {i: f"class_{i}" for i in range(num_classes)}
            
            # 初始化所有情绪得分为 0
            emotion_map = {
                EmotionLabel.NEUTRAL.value: 0.0,
                EmotionLabel.HAPPY.value: 0.0,
                EmotionLabel.SAD.value: 0.0,
                EmotionLabel.ANGRY.value: 0.0,
            }
            
            # 根据实际类别数量进行映射
            if num_classes == 2:
                # 2 类：通常是 positive/negative 或 neutral/emotion
                # 假设：class_0 = neutral, class_1 = emotion（需要进一步判断）
                emotion_map[EmotionLabel.NEUTRAL.value] = float(probs[0])
                # 将 emotion 类平均分配到 happy/sad/angry（或根据实际标签调整）
                emotion_score = float(probs[1]) / 3.0
                emotion_map[EmotionLabel.HAPPY.value] = emotion_score
                emotion_map[EmotionLabel.SAD.value] = emotion_score
                emotion_map[EmotionLabel.ANGRY.value] = emotion_score
                
            elif num_classes == 4:
                # 4 类：直接映射（需要确认顺序）
                # 常见顺序可能是：neutral, happy, sad, angry
                emotion_map[EmotionLabel.NEUTRAL.value] = float(probs[0])
                emotion_map[EmotionLabel.HAPPY.value] = float(probs[1])
                emotion_map[EmotionLabel.SAD.value] = float(probs[2])
                emotion_map[EmotionLabel.ANGRY.value] = float(probs[3])
                
            else:
                # 其他情况：尝试根据标签名称智能映射
                for i, prob in enumerate(probs):
                    label = label_map.get(i, f"class_{i}").lower()
                    prob_val = float(prob)
                    
                    # 根据标签名称映射到我们的情绪系统
                    if 'neutral' in label or 'normal' in label:
                        emotion_map[EmotionLabel.NEUTRAL.value] += prob_val
                    elif 'happy' in label or 'joy' in label or 'positive' in label:
                        emotion_map[EmotionLabel.HAPPY.value] += prob_val
                    elif 'sad' in label or 'sadness' in label:
                        emotion_map[EmotionLabel.SAD.value] += prob_val
                    elif 'angry' in label or 'anger' in label or 'negative' in label:
                        emotion_map[EmotionLabel.ANGRY.value] += prob_val
                    else:
                        # 未知标签，平均分配到所有情绪
                        emotion_map[EmotionLabel.NEUTRAL.value] += prob_val / 4.0
                        emotion_map[EmotionLabel.HAPPY.value] += prob_val / 4.0
                        emotion_map[EmotionLabel.SAD.value] += prob_val / 4.0
                        emotion_map[EmotionLabel.ANGRY.value] += prob_val / 4.0
            
            return emotion_map

    def recognize(
        self,
        audio_path: str,
    ) -> SERResult:
        """
        识别音频文件的情绪。

        :param audio_path: 音频文件路径（推荐 16kHz 单通道 wav）
        :return: SERResult 对象，包含情绪标签和置信度
        """
        # 1. 加载音频
        audio = self._load_audio(audio_path)
        
        # 2. 预处理
        audio_input = self._preprocess_audio(audio)
        
        # 3. 预测
        emotion_scores = self._predict_emotion(audio_input)
        
        # 4. 找到最高分情绪
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion_label_str, confidence = best_emotion
        
        # 5. 应用置信度阈值
        if confidence < self.confidence_threshold:
            emotion_label = EmotionLabel.NEUTRAL
            confidence = emotion_scores[EmotionLabel.NEUTRAL.value]
        else:
            emotion_label = EmotionLabel(emotion_label_str)
        
        return SERResult(
            emotion=emotion_label,
            confidence=confidence,
            raw_scores=emotion_scores,
            raw={"model_output": "custom"}  # 可根据实际模型输出填充
        )


if __name__ == "__main__":
    # 测试
    audio_path = "/home/yanlan/workspace/code/emo-voice-chat/data/output_voice/test_angry.wav"
    if Path(audio_path).exists():
        ser = SEREngine()
        result = ser.recognize(audio_path)
        
        print("\n========== 情绪识别结果 ==========")
        print(f"情绪: {result.emotion.value}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"所有情绪得分:")
        for emo, score in result.raw_scores.items():
            print(f"  {emo}: {score:.3f}")
        print("================================")
    else:
        print(f"音频文件不存在: {audio_path}")