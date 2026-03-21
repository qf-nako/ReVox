import cv2
import numpy as np
import os

def enhance_video(input_path, output_path, scale=2, method='cubic'):
    """
    对视频进行超分辨率放大（插值方式）。

    Args:
        input_path (str): 输入视频路径。
        output_path (str): 输出视频路径。
        scale (int): 放大倍数，例如 2 表示放大到 2 倍尺寸。
        method (str): 插值方法，'cubic' (双三次) 或 'linear' (双线性) 或 'lanczos' (Lanczos)。
                      默认 'cubic' 效果较好。

    Returns:
        str: 输出视频路径。
    """
    # 映射插值方法到 OpenCV 常量
    inter_methods = {
        'cubic': cv2.INTER_CUBIC,
        'linear': cv2.INTER_LINEAR,
        'lanczos': cv2.INTER_LANCZOS4
    }
    inter = inter_methods.get(method.lower(), cv2.INTER_CUBIC)

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_width = width * scale
    new_height = height * scale

    # 视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 放大帧
        enlarged = cv2.resize(frame, (new_width, new_height), interpolation=inter)
        out.write(enlarged)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"超分辨率进度: {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    return output_path


def enhance_video_dnn(input_path, output_path, scale=2, model_path=None, model_name='espcn'):
    """
    使用 DNN 模型进行超分辨率（需要预训练模型）。
    此函数为扩展预留，当前未实现模型自动下载。
    """
    # 如需实现，可参考 OpenCV DNN 超分示例
    # 这里简单抛出未实现异常，用户可后续完善
    raise NotImplementedError("DNN 超分需要额外模型，请先下载模型文件。使用 enhance_video 简单插值版本。")