import os, math, torch, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from cnn_model.cnn_model import VGG16
from vision_transformer.vit_model import vit_base_patch32_224
from vision_transformer.vit_model import vit_base_patch16_224
from hybrid_model_utils import read_split_data, train_one_epoch, evaluate

class hybrid_model(nn.Module):
    def __init__(self, Transformer, VGG16, batch_size, device):
        super(hybrid_model, self).__init__()
        self.model1 = nn.DataParallel(Transformer().to(device), device_ids=[0, 1])
        # self.model1 = Transformer()
        self.linear1 = nn.Linear(1000, 500).to(device)
        self.model2 = nn.DataParallel(VGG16().to(device), device_ids=[0, 1])
        # self.model2 = VGG16()
        self.linear2 = nn.Linear(batch_size, 1).to(device)

    def forward(self, x1, x2):

        x1 = self.model1(x1)
        x1 = self.linear1(x1)

        x2 = self.model2(x2)
        x2 = self.linear1(x2)
        x2 = x2.transpose(0, 1)
        # x2 = x2.unsqueeze(-1)

        x3 = torch.matmul(x1, x2)
        x3 = self.linear2(x3).squeeze(1)

        # x3 = torch.softmax(x3, dim=-1) # softmax + cross
        # x3 = torch.sigmoid(x3) # simoid + BCEloss
        return x3

class VitDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class VggDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("././weights") is False:
        os.makedirs("././weights")

    tb_writer = SummaryWriter("././runs/police/")

    train_tras_path, train_tras_label, val_tras_path, val_tras_label = read_split_data(
        "././dataset/police_trajectory")
    train_imgs_path, train_imgs_label, val_imgs_path, val_imgs_label = read_split_data(
        "././dataset/bboxes_frame_label")

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

    batch_size = args.batch_size
    # nw = 0
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    Vit_train_dataset = VitDataSet(images_path=train_tras_path,
                                   images_class=train_tras_label,
                                   transform=data_transform["train"])
    Vit_val_dataset = VitDataSet(images_path=val_tras_path,
                                 images_class=val_tras_label,
                                 transform=data_transform["val"])

    Vgg_train_dataset = VggDataSet(images_path=train_imgs_path,
                                   images_class=train_imgs_label,
                                   transform=data_transform["train"])
    Vgg_val_dataset = VggDataSet(images_path=val_imgs_path,
                                 images_class=val_imgs_label,
                                 transform=data_transform["val"])

    Vit_train_loader = DataLoader(Vit_train_dataset, pin_memory=True, num_workers=nw)
    Vit_val_loader = DataLoader(Vit_val_dataset, pin_memory=True, num_workers=nw)
    Vgg_train_loader = DataLoader(Vgg_train_dataset, pin_memory=True, num_workers=nw)
    Vgg_val_loader = DataLoader(Vgg_val_dataset, pin_memory=True, num_workers=nw)

    label, imgs, tras = map(dataloader_sort, [Vgg_train_loader, Vgg_train_loader, Vit_train_loader], [1, 0, 0],
                            [True, False, False])
    label_v, imgs_v, tras_v = map(dataloader_sort, [Vgg_val_loader, Vgg_val_loader, Vit_val_loader], [1, 0, 0],
                                  [True, False, False])

    # 整合两种数据集 DataLoader[0]为轨迹, [1]当前帧, [2]标签 (数据集做好的情况下两者标签应为一致)
    train_loader = DataLoader(TensorDataset(tras, imgs, label),
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=nw,
                              shuffle=True)

    val_loader = DataLoader(TensorDataset(tras_v, imgs_v, label_v),
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=nw,
                            shuffle=True)

    Vit = vit_base_patch32_224
    CNN = VGG16
    model = hybrid_model(Vit, CNN, args.batch_size, device=device)
    # model = nn.DataParallel(model.to(device), device_ids=None)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=0)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):

        torch.cuda.empty_cache()

        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)
        scheduler.step()
        val_loss, val_acc, TP, FN, FP, TN = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "TP", "FN", "FP", "TN"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], TP, epoch)
        tb_writer.add_scalar(tags[6], FN, epoch)
        tb_writer.add_scalar(tags[7], FP, epoch)
        tb_writer.add_scalar(tags[8], TN, epoch)

        torch.save(model.state_dict(), "././weights/hybrid_model_police.pkl")

def dataloader_sort(loader, index, is_label):
    if not is_label:
        return torch.from_numpy(np.array([i[index][0].numpy().tolist() for i in iter(loader)])).float()
    else:
        return torch.from_numpy(np.array([i[index][0].numpy().tolist() for i in iter(loader)])).long()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--weights', type=str, default='./vision_transformer/swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--device', default='cuda', help='device id (i.e.'' 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)