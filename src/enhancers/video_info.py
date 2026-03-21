import cv2
import os
import json

def extract_video_info(video_path, save_json=None):
    """
    提取视频文件的基本信息。

    Args:
        video_path (str): 视频文件路径。
        save_json (str, optional): 如果提供，将信息保存为 JSON 文件。

    Returns:
        dict: 包含视频信息的字典。
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    # 获取基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])  # 四字符代码
    file_size = os.path.getsize(video_path)

    info = {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "codec": codec,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2)
    }

    cap.release()

    if save_json:
        with open(save_json, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"视频信息已保存至: {save_json}")

    return info


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) > 1:
        info = extract_video_info(sys.argv[1])
        print(json.dumps(info, indent=4))
    else:
        print("用法: python video_info.py <视频文件>")