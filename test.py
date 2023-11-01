import torch, cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from hybrid_model import hybrid_model
from cnn_model.cnn_model import VGG16
from vision_transformer.vit_model import vit_base_patch32_224
import matplotlib.pyplot as plt


def preict_one_img(img_path, model_path):
    Vit = vit_base_patch32_224
    CNN = VGG16
    net = hybrid_model(Vit, CNN, 8, device=device).to(device)
    net.load_state_dict(torch.load(model_path))
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    # 1.将numpy数据变成tensor
    tran = transforms.ToTensor()
    img = tran(img)
    img = img.to(device)
    # 2.将数据变成网络需要的shape
    img = img.view(1, 3, 224, 224)

    out1 = net(img)
    out1 = F.softmax(out1, dim=1)
    proba, class_ind = torch.max(out1, 1)




if __name__ == '__main__':
    classes = ["Accident", "No_accident"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = r"C:\Users\YIHANG\PycharmProjects\AutomaticScenarioGenerationSystem\input\input_frame/0001/0001_00400_2D.png"
    model_path = "weights/hybrid_model.pkl"
    preict_one_img(img_path, model_path)
