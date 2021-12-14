from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import codecs
from PIL import Image
from PIL import ImageChops
import torchvision.transforms as transforms
import numpy as np
from models import MNIST_target_net
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torch.utils.data import Dataset

use_cuda=True


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./MNIST_target_F1_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()


datapath = './adv_dir'
txtpath = './label.txt'


class MyDataset(Dataset):
    def __init__(self,txtpath):
        imgs = []
        datainfo = open(txtpath,'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'/'+pic)
        pic = pic.convert("L")
        pic = transforms.ToTensor()(pic)
        #label = transforms.ToTensor()(label)
        return pic,label
        

data = MyDataset(txtpath)
train_dataloader = DataLoader(data,batch_size=1,shuffle=True,num_workers=12)


num_correct = 0
for i, data in enumerate(train_dataloader, 0):
  test_img, test_label = data
  test_img, test_label = test_img.to(device), test_label
  pred_lab = torch.argmax(target_model(test_img), 1)

  for j in range(len(test_label)):
    if (pred_lab.item()==int(test_label[j])):
      num_correct+=1
  #num_correct += torch.sum(pred_lab==test_label,0)
  #print(pred_lab.item(),test_label[j])

final_acc = num_correct/float(len(train_dataloader))
print("Test Accuracy = {} / {} = {}".format(num_correct, len(train_dataloader), final_acc))
