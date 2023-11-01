import matplotlib.pyplot as plt
import csv

'''读取csv文件, 画图前需要删除csv中的第一行'''

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y

# loss
def loss_plt():
    plt.figure(1)
    x1, y1 = readcsv('./graduate/train_loss.csv')
    plt.plot(x1, y1, color='red', label='train_loss')
    x2, y2 = readcsv('./graduate/val_loss.csv')
    plt.plot(x2, y2, color='g', label='valid_loss')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    # plt.title('The loss of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig('./graduate/loss.jpg')
    plt.show()

# acc
def acc_plt():
    plt.figure(2)
    x3, y3 = readcsv('./graduate/train_acc.csv')
    plt.plot(x3, y3, color='red', label='train_acc')
    x4, y4 = readcsv('./graduate/val_acc.csv')
    plt.plot(x4, y4, color='g', label='valid_acc')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    # plt.title('The accuracy of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig('./graduate/acc.jpg')
    plt.show()

if __name__ == '__main__':
    loss_plt()
    acc_plt()