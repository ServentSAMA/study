'''
ffmpeg 批量转换视频文件到mp4格式
'''
import os
import subprocess

def ffmpeg_test():
    # 批量转换
    for file in os.listdir('.'):
        if file.endswith('.mp4'):
            cmd = ["ffmpeg", "-i", file, "/home/wuniu/video/" + file, "-y"]
            subprocess.call(cmd)


