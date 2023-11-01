import cv2

video_name = "000567.mp4"
vidcap = cv2.VideoCapture(r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\SCI3\video_yolo/%s' % video_name)
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    cv2.imwrite(r"C:\Users\YIHANG\PycharmProjects\HTAD_dataset\SCI3\video_frame/" + video_name[:6] + "_%d.png" %count, image)
    if cv2.waitKey(10) == 27:
        break
    count += 1