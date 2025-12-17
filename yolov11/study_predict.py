import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0", show=True)


# for result in results:
#     plotted_img = result.plot()
#     cv2.imshow('Detection Result', plotted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("8bde04797b2ff89491f4da6c84936846.jpg")
results = model.predict(source=im1, save=False)  # save plotted images
print(results)
# View results
for r in results:
    if r.boxes.cls[0] == 5:
        x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        orig_img = r.orig_img
        print(x1, y1, x2, y2)
        # 裁剪图像
        cropped_img = orig_img[y1:y2, x1:x2]

        # 保存或显示
        cv2.imwrite('bus_cropped_img.jpg', cropped_img)

      # print the Boxes object containing the detection bounding boxes

# from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])