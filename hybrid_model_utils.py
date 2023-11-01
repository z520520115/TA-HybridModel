import os
import sys
import json
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    # print("dataset load")

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def list_compared(list1, list2):
    TP, FN, FP, TN = 0, 0, 0, 0
    for idx1, preds in enumerate(list1):
        for idx2, labels in enumerate(list2):
            if idx1 == idx2:
                if preds == labels == 0:
                    TP += 1
                if preds == labels == 1:
                    TN += 1
                if preds == 0 and labels == 1:
                    FP += 1
                if preds == 1 and labels == 0:
                    FN += 1
    return TP, FN, FP, TN

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.BCEWithLogitsLoss()
    accu_loss = 0 # 累计损失
    accu_num = 0 # 累计预测正确的样本数

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout) # 进度条

    for step, data in enumerate(data_loader):
        tras, imgs, labels = data
        sample_num += imgs.shape[0]

        imgs = imgs.to(device).float()
        tras = tras.to(device).float()

        pred = model(imgs, tras)
        pred_classes = torch.LongTensor([1 if _ > 0.5 else 0 for _ in pred]).to(device)
        # pred_classes = torch.max(pred, dim=0)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum().item()

        loss = loss_function(pred.float(), labels.to(device).float()) # BCEloss最后传到crossentropyloss中需要为float型
        loss.backward()
        accu_loss += loss.item()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss/ (step + 1),
                                                                                accu_num / sample_num)

        # print("[train epoch {}] loss: {}, acc: {}".format(epoch, float(accu_loss.item() / (step + 1)), float(accu_num.item()/sample_num)))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss/ (step + 1), accu_num / sample_num


def evaluate(model, data_loader, device, epoch):

    model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.BCEWithLogitsLoss()

    accu_num = 0   # 累计预测正确的样本数
    accu_loss = 0  # 累计损失
    TP, FN, FP, TN = 0, 0, 0, 0

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        tras, imgs, labels = data
        sample_num += imgs.shape[0]

        imgs = imgs.to(device).float()
        tras = tras.to(device).float()

        pred = model(imgs, tras)
        pred_classes = torch.LongTensor([1 if _ > 0.5 else 0 for _ in pred]).to(device)
        TP_bs, FN_bs, FP_bs, TN_bs = list_compared(pred_classes.cpu().numpy().tolist(), labels.cpu().numpy().tolist())
        TP += TP_bs
        FN += FN_bs
        FP += FP_bs
        TN += TN_bs
        accu_num += torch.eq(pred_classes, labels.to(device)).sum().item()

        loss = loss_function(pred.float(), labels.to(device).float())
        accu_loss += loss.item()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss/ (step + 1),
                                                                               accu_num / sample_num)
        # print("[valid epoch {}] loss: {}, acc: {}".format(epoch, float(accu_loss.item() / (step + 1)), float(accu_num.item() / sample_num)))
    return accu_loss / (step + 1), accu_num / sample_num, TP, FN, FP, TN
