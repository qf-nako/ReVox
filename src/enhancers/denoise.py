import os
import noisereduce as nr
import soundfile as sf


def process_audio(input_path, output_path=None, prop_decrease=0.8, **kwargs):
    """
    对音频文件进行降噪处理。

    Args:
        input_path (str): 输入音频文件的路径。
        output_path (str, optional): 输出音频文件的路径。如果为 None，则在输入文件同目录下创建
                                      一个带有 '_denoised' 后缀的文件。
        prop_decrease (float): 降噪强度，取值范围 [0, 1]。1.0 表示最大降噪。
        **kwargs: 传递给 nr.reduce_noise 的其他参数（如 stationary、n_std_thresh_stationary 等）。

    Returns:
        str: 降噪后的音频文件路径。

    Raises:
        FileNotFoundError: 如果输入文件不存在。
        RuntimeError: 降噪过程中的错误。
    """
    # 检查输入文件是否存在
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"音频文件不存在: {input_path}")

    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised{ext}"
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    try:
        # 读取音频
        data, sr = sf.read(input_path)
        # 执行降噪
        reduced_noise = nr.reduce_noise(
            y=data,
            sr=sr,
            prop_decrease=prop_decrease,
            **kwargs
        )
        # 保存结果
        sf.write(output_path, reduced_noise, sr)
        print(f"降噪完成，输出文件: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"音频降噪失败: {e}")


# 简单的自测（当直接运行此文件时执行）
if __name__ == "__main__":
    # 测试文件路径，可以修改为实际存在的音频文件
    test_input = "./examples/LipTest.wav"
    if os.path.exists(test_input):
        process_audio(test_input)
    else:
        print("测试音频不存在，请指定有效路径。")
        # 可以手动输入路径测试
        # path = input("请输入音频文件路径: ")
        # process_audio(path)
