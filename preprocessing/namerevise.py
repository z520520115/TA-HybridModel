import os


path = (r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\SCI3/video_frame_boundingbox_new')

for root, dirs, files in os.walk(path):
    for i in files:
        f_path = (path + '/' + i)
        oldname = f_path
        newname = f_path[:-4] + '_mask.jpg'
        os.rename(oldname, newname)
