import os, torch, argparse, sys
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 输入是224*224*3 计算(224-5)/1+1=220 conv1输出的结果是220
        self.conv1 = nn.Conv2d(3, 6, 5)  # input:3 output6 kernel:5
        # 输入是220*220*6 窗口2*2  计算(220-0)/2=110 通过max_pooling层输出的是110*110*6
        self.pool = nn.MaxPool2d(2, 2)
        # 输入是220*220*6，计算（110 - 5）/ 1 + 1 = 106，通过conv2输出的结果是106*106*16
        self.conv2 = nn.Conv2d(6, 16, 5)  # input:6, output:16, kernel:5

        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        x = x.view(-1, 16 * 53 * 53)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def train(model, optimizer, data_loader, device, epoch):

    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = 0
    accu_num = 0
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        tras, labels = data
        sample_num += tras.shape[0]

        tras = tras.to(device)
        labels = labels.to(device)

        pred = model(tras)
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels).sum().item()

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.item()

        # output = cnn_model(imgs)
        # train_predict = torch.max(output.data, 1)[1]

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss / (step + 1),
                                                                               accu_num / sample_num)
        optimizer.step()
        optimizer.zero_grad()

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss / (step + 1), accu_num / sample_num

def evaluate(model, data_loader, device, epoch):

    model.eval()

    loss_function = torch.nn.CrossEntropyLoss()

    accu_num = 0
    accu_loss = 0
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        tras, labels = data
        sample_num += tras.shape[0]

        tras = tras.to(device)
        labels = labels.to(device)

        pred = model(tras)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum().item()

        loss = loss_function(pred, labels)
        accu_loss += loss.item()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss / (step + 1),
                                                                               accu_num / sample_num)
    return accu_loss / (step + 1), accu_num / sample_num

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    tb_writer = SummaryWriter("./runs/compare_cnn/")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 12])

    pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    amount_ac = len(os.listdir(args.dataset_dir + '/accident'))
    amount_nac = len(os.listdir(args.dataset_dir + '/no_accident'))
    print('accident images : %d, no_accident images : %d' % (amount_ac, amount_nac))

    train_dataset = datasets.ImageFolder(root=args.dataset_dir, transform=pipeline)
    train_loader = DataLoader(train_dataset, batch_size=amount_ac + amount_nac)

    images, labels = next(iter(train_loader))
    x_train, y_train, x_label, y_label = train_test_split(images, labels, test_size=0.2)
    train_loader = DataLoader(TensorDataset(x_train, x_label), batch_size=args.batch_size,
                              pin_memory=True, num_workers=nw, shuffle=True)
    val_loader = DataLoader(TensorDataset(y_train, y_label), batch_size=args.batch_size,
                            pin_memory=True, num_workers=nw, shuffle=True)

    model = CNN_Model().to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        torch.cuda.empty_cache()
        train_loss, train_acc = train(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch)
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # torch.save(model.state_dict(), "model/sci_cnn.pth")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset-dir', type=str, default='./dataset/trajectory_mask_100')

    opt = parser.parse_args()

    main(opt)