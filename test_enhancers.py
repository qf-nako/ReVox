import sys
import traceback

sys.path.insert(0, 'src')
print("尝试导入 enhancers...")

try:
    import enhancers
    print("成功导入 enhancers 包")
except Exception as e:
    print("导入 enhancers 包失败:")
    traceback.print_exc()

print("\n尝试从 enhancers.denoise 导入 process_audio...")
try:
    from enhancers.denoise import process_audio
    print("成功导入 process_audio")
except Exception as e:
    print("导入 process_audio 失败:")
    traceback.print_exc()

print("\n尝试从 enhancers.static_bg 导入 extract_static_background...")
try:
    from enhancers.static_bg import extract_static_background
    print("成功导入 extract_static_background")
except Exception as e:
    print("导入 extract_static_background 失败:")
    traceback.print_exc()