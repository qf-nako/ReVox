import sys
import os
# 将当前文件所在目录的父目录（即项目根目录）加入到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import os
import soundfile as sf
import numpy as np
from src.utils.audio_utils import preprocess_audio
from src.enhancers.denoise import process_audio


class TestAudioPipeline(unittest.TestCase):
    def setUp(self):
        # 创建一个临时的测试音频文件 (1秒的正弦波 + 随机噪声)
        self.test_file = "test_input.wav"
        self.sample_rate = 44100  # 使用非标准采样率测试重采样
        t = np.linspace(0, 1, self.sample_rate)
        # 生成 440Hz 正弦波 + 噪声
        signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(self.sample_rate)
        sf.write(self.test_file, signal, self.sample_rate)

        self.output_std = "test_std.wav"
        self.output_denoised = "test_denoised.wav"

    def tearDown(self):
        # 测试结束清理文件
        for f in [self.test_file, self.output_std, self.output_denoised]:
            if os.path.exists(f):
                os.remove(f)

    def test_01_preprocessing(self):
        """测试音频重采样和标准化"""
        print("\n测试音频标准化 (audio_utils)...")
        out = preprocess_audio(self.test_file, self.output_std, target_sr=16000)
        self.assertTrue(os.path.exists(out))

        # 验证采样率是否正确
        data, sr = sf.read(out)
        self.assertEqual(sr, 16000)
        print(f"采样率已成功转换: {sr}Hz")

    def test_02_denoising(self):
        """测试音频降噪"""
        print("\n测试音频降噪 (denoise)...")
        # 先标准化
        std_audio = preprocess_audio(self.test_file, self.output_std, target_sr=16000)
        # 再降噪
        out = process_audio(std_audio, self.output_denoised, prop_decrease=0.9)

        self.assertTrue(os.path.exists(out))

        # 验证降噪后能量是否减小（由于噪声被移除，方差通常会变小）
        original_data, _ = sf.read(std_audio)
        denoised_data, _ = sf.read(out)
        self.assertLess(np.var(denoised_data), np.var(original_data))
        print(f"[V] 降噪测试通过，能量分布已改变。")


if __name__ == '__main__':
    unittest.main()