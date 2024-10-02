import cv2
import numpy as np

# error: (-215:Assertion failed) size.width>0 && size.height>0 可修改视频路径
cap = cv2.VideoCapture("E:\\电影\\冰雪奇缘.特效中英字幕.Frozen.2013.1080P.X264.AAC.Englishi&Mandarin&Taiwanese&Cantonese.CHS-ENG.FFans\\冰雪奇缘.特效中英字幕.Frozen.2013.1080P.X264.AAC.Englishi&Mandarin&Taiwanese&Cantonese.CHS-ENG.FFans.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
while (True):
    # 读帧
    ret, frame = cap.read()
    # 显示图像
    cv2.imshow("video", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break
# 释放cap
cap.release()
# 关闭窗口，清除程序所占用的内存
cv2.destroyAllWindows()


