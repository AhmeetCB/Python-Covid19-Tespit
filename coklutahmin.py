import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import argparse
import glob
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str)
parser.add_argument('--trained_model_path', type=str, default='./coronaPt_50.pt')
parser.add_argument('--cut_off_threshold', type=float, default= 0.2)
parser.add_argument('--batch_size', type=int, default=20)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['covid','non']

model_name= args.trained_model_path
model= torch.load(model_name, map_location='cpu') 
model.eval()

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

imsize= 224
loader = transforms.Compose([transforms.Resize(imsize), 
                             transforms.CenterCrop(224), 
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])

sm = torch.nn.Softmax(dim=1)
test_covid  =glob.glob("%s*" %args.folder_path)
covid_pred= np.zeros([len(test_covid),1]).astype(int)
covid_prob= np.zeros([len(test_covid),1])
for i in range(len(test_covid)):
    cur_img= image_loader(test_covid[i])
    model_output= model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]
    cur_prob = sm(model_output)
    covid_prob= cur_prob.data.numpy()[0,0]
    print("%03d. fotoğrafın tahmini %s ihtimalle %s" %(i,covid_prob,class_names[int(cur_pred.data.numpy())]))