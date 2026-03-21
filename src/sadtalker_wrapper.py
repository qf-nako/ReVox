import os
import subprocess
import sys
import glob
import shutil
import time
from enhancers.denoise import process_audio
from enhancers.superres import enhance_video
from enhancers.video_info import extract_video_info

def copy_audio(video_without_audio, original_video, output_video):
    """
    将 original_video 的音频流复制到 video_without_audio，生成 output_video。
    要求 ffmpeg 在系统 PATH 中。
    """
    cmd = [
        'ffmpeg', '-i', video_without_audio,
        '-i', original_video,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', output_video
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_video
    except subprocess.CalledProcessError as e:
        print(f"音频合并失败: {e.stderr}")
        return None

def run_sadtalker(source_image, driven_audio, output_path, sadtalker_path=None,
                  enhancer=None, denoise=False, superres=False, superres_scale=2,
                  extract_info=False, info_json=None,
                  sadtalker_python=None, **kwargs):
    """
    调用 SadTalker 生成视频，可选择启用：
      - 音频降噪 (denoise)
      - 超分辨率放大 (superres)
      - 视频信息提取 (extract_info)
    """
    # 自动定位 SadTalker 目录
    if sadtalker_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.abspath(os.path.join(script_dir, '..', '..', 'SadTalker'))
        if not os.path.isdir(candidate):
            candidate2 = os.path.abspath(os.path.join(script_dir, '..', 'SadTalker'))
            if os.path.isdir(candidate2):
                sadtalker_path = candidate2
            else:
                raise FileNotFoundError("无法自动定位 SadTalker 目录，请通过 sadtalker_path 指定。")
        else:
            sadtalker_path = candidate
    sadtalker_path = os.path.abspath(sadtalker_path)

    # Python 解释器：优先使用原环境（sadtalker），确保 GPU 可用
    if sadtalker_python is None:
        candidate_python = r"G:\conda_envs\sadtalker\python.exe"
        if os.path.isfile(candidate_python):
            sadtalker_python = candidate_python
            print(f"使用原环境 Python: {sadtalker_python}")
        else:
            sadtalker_python = sys.executable
            print(f"使用当前环境 Python: {sadtalker_python}")

    # 检查必需文件
    inference_script = os.path.join(sadtalker_path, "inference.py")
    if not os.path.isfile(inference_script):
        raise FileNotFoundError(f"inference.py 不存在: {inference_script}")

    checkpoint_dir = os.path.join(sadtalker_path, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.join(sadtalker_path, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError("模型目录不存在，请确保 SadTalker 模型已下载。")

    # 输入路径绝对化
    source_image = os.path.abspath(source_image)
    driven_audio = os.path.abspath(driven_audio)
    if not os.path.isfile(source_image):
        raise FileNotFoundError(f"源图片不存在: {source_image}")
    if not os.path.isfile(driven_audio):
        raise FileNotFoundError(f"驱动音频不存在: {driven_audio}")

    output_path = os.path.abspath(output_path)
    result_dir = os.path.dirname(output_path)
    os.makedirs(result_dir, exist_ok=True)

    # ---------- 1. 音频降噪 ----------
    temp_files = []
    if denoise:
        print("启用音频降噪预处理...")
        audio_basename = os.path.basename(driven_audio)
        denoised_audio = os.path.join(result_dir, f"temp_denoised_{audio_basename}")
        driven_audio = process_audio(driven_audio, output_path=denoised_audio, prop_decrease=0.8)
        temp_files.append(denoised_audio)

    # ---------- 2. 调用 SadTalker 生成带音频的视频 ----------
    cmd = [
        sadtalker_python,
        inference_script,
        "--source_image", source_image,
        "--driven_audio", driven_audio,
        "--result_dir", result_dir,
        "--checkpoint_dir", checkpoint_dir,
    ]
    if enhancer:
        cmd.extend(["--enhancer", enhancer])
    for key, value in kwargs.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value not in (False, None):
            cmd.append(f"--{key}")
            cmd.append(str(value))

    before_items = set(os.listdir(result_dir))
    my_env = os.environ.copy()
    my_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        subprocess.run(cmd, check=True, cwd=sadtalker_path, env=my_env)
    except subprocess.CalledProcessError as e:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        raise RuntimeError(f"SadTalker 调用失败: {e}")

    time.sleep(2)
    after_items = set(os.listdir(result_dir))
    new_items = after_items - before_items
    video_candidates = []

    for item in new_items:
        item_path = os.path.join(result_dir, item)
        if os.path.isfile(item_path) and item_path.lower().endswith('.mp4'):
            video_candidates.append(item_path)
        elif os.path.isdir(item_path):
            sub_files = glob.glob(os.path.join(item_path, "**", "*.mp4"), recursive=True)
            video_candidates.extend(sub_files)

    if not video_candidates:
        video_candidates = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)

    if not video_candidates:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        raise FileNotFoundError(f"在 {result_dir} 中未找到生成的视频文件")

    generated_video = max(video_candidates, key=os.path.getmtime)
    if os.path.abspath(generated_video) != os.path.abspath(output_path):
        shutil.move(generated_video, output_path)

    # ---------- 3. 超分辨率放大（并恢复音频） ----------
    if superres:
        print(f"启用超分辨率放大（{superres_scale}x）...")
        temp_sr = output_path + ".sr.mp4"
        try:
            # 先放大帧（无声音）
            enhance_video(output_path, temp_sr, scale=superres_scale, method='cubic')
            # 将原始视频的音频合并回去
            if os.path.exists(temp_sr):
                temp_with_audio = output_path + ".sr_with_audio.mp4"
                if copy_audio(temp_sr, output_path, temp_with_audio):
                    os.replace(temp_with_audio, output_path)
                    print("超分辨率完成，已合并音频")
                else:
                    # 回退到无音频版本
                    os.replace(temp_sr, output_path)
                    print("超分辨率完成，但音频合并失败，结果无声音")
            else:
                raise Exception("超分视频未生成")
        except Exception as e:
            print(f"超分辨率失败: {e}，将使用原始视频。")
            if os.path.exists(temp_sr):
                os.remove(temp_sr)

    # ---------- 4. 视频信息提取 ----------
    if extract_info:
        print("提取视频信息...")
        try:
            info = extract_video_info(output_path, save_json=info_json)
            print("视频信息:")
            for key, val in info.items():
                print(f"  {key}: {val}")
        except Exception as e:
            print(f"信息提取失败: {e}")

    # 清理临时音频文件
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"已清理临时文件: {f}")

    # 清理空文件夹
    for item in new_items:
        item_path = os.path.join(result_dir, item)
        if os.path.isdir(item_path):
            try:
                if not os.listdir(item_path):
                    os.rmdir(item_path)
                    print(f"已删除空目录: {item_path}")
            except Exception:
                pass

    for root, dirs, files in os.walk(result_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass

    return output_path


if __name__ == "__main__":
    run_sadtalker(
        source_image=r".\examples\man.png",
        driven_audio=r".\examples\LipTest.wav",
        output_path="./examples/test_output_final.mp4",
        enhancer=None,
        still=True,
        denoise=True,
        superres=True,
        superres_scale=2,
        extract_info=True,
        info_json="./examples/test_info.json"
    )