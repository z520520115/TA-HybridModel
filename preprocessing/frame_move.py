import os
from shutil import copy

'''
批量移动视频帧, 原数据集为(0-52), 忽略0和52帧, 将1-25送入no_accident文件夹, 26-50送入accident文件夹
'''

path =  r'C:\Users\YIHANG\Desktop\CADP\train/'
accident_path = r'C:\Users\YIHANG\PycharmProjects\folwnet2\flownet2-pytorch\datasets\video_frame_150\train\accident/'
no_accident_path = r'C:\Users\YIHANG\PycharmProjects\folwnet2\flownet2-pytorch\datasets\video_frame_150\train\no_accident/'
# noneed_path = r'C:\Users\YIHANG\PycharmProjects\HTAD_dataset\current_frame\no_need/'

# 列表元素str转int
def str_to_int(list):
    new_list = []
    for i in list:
        new_list.append(int(i[7:-4]))
    list = new_list
    return list

# 列表元素str转int
def int_to_str(list, file_name):
    new_list = []
    for i in list:
        new_list.append(file_name + '_' + str(i) + ".jpg")
    list = new_list
    return list

# 遍历全frames路径
for root, dirs, files in os.walk(path):
    for dir_idx, j in enumerate(dirs):

        # 添加每个子文件夹的path
        path_f = path + '/' + j
        # 遍历子文件夹所有frames
        for root, dirs, file in os.walk(path_f):

            # 字符串转整形→排序→整形转字符串
            file = str_to_int(file)
            file.sort()
            file = int_to_str(file, j)

            # 复制文件到新地址
            for idx, i in enumerate(file):
                old_path = path_f + '/' + i
                if -1 < idx < 25:  # (2-26)
                    copy(old_path, no_accident_path)
                elif 24 < idx < 50: # (27-51)
                    copy(old_path, accident_path)
                # else:
                #     copy(old_path, noneed_path)