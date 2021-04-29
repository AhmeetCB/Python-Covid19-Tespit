from __future__ import print_function
import torch, os, copy, time, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import glob, pickle
import seaborn as sn
import argparse
start_time= time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--test_covid_path', type=str, default='./data/test/covid/')
parser.add_argument('--test_non_covid_path', type=str, default='./data/test/non/')
parser.add_argument('--trained_model_path', type=str, default='./coronaPt_50.pt')
parser.add_argument('--cut_off_threshold', type=float, default= 0.2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()
def find_sens_spec( covid_prob, noncovid_prob, thresh):
    duyarlilik= (covid_prob >= thresh).sum()   / (len(covid_prob)+1e-10)
    ozgulluk= (noncovid_prob < thresh).sum() / (len(noncovid_prob)+1e-10)
    print("Duyarlılık= %.3f, Özgüllük= %.3f" %(duyarlilik,ozgulluk))
    return duyarlilik, ozgulluk
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['covid','non']

model_name= args.trained_model_path
model= torch.load(model_name, map_location='cpu') 
model.eval()

imsize= 224
loader = transforms.Compose([transforms.Resize(imsize),transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

sm = torch.nn.Softmax(dim=1)

#Bütün verilerin tahmini
test_covid  = glob.glob("%s*" %args.test_covid_path)
test_non    = glob.glob("%s*" %args.test_non_covid_path)

covid_pred= np.zeros([len(test_covid),1]).astype(int)
non_pred  = np.zeros([len(test_non),1]).astype(int)

covid_prob= np.zeros([len(test_covid),1])
non_prob   = np.zeros([len(test_non),1])

for i in range(len(test_covid)):
    cur_img= image_loader(test_covid[i])
    model_output= model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]
    cur_prob = sm(model_output)
    covid_prob[i,:]= cur_prob.data.numpy()[0,0]
    print("%03d. Covid olan fotoğrafın tahmini:%s" %(i, class_names[int(cur_pred.data.numpy())]) )

for i in range(len(test_non)):
    cur_img= image_loader(test_non[i])
    model_output= model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]  
    cur_prob = sm(model_output)
    non_prob[i,:]= cur_prob.data.numpy()[0,0]
    print("%03d. Covid olmayan fotografın tahmini:%s" %(i, class_names[int(cur_pred.data.numpy())]) )

thresh= args.cut_off_threshold
sensitivity_40, specificity= find_sens_spec( covid_prob, non_prob, thresh)

covid_pred = np.where( covid_prob  >thresh, 1, 0)
non_pred   = np.where( non_prob  >thresh,   1, 0)

############### Hata Matrisi oluşturma
covid_list= [int(covid_pred[i]) for i in range(len(covid_pred))]
covid_count = [(x, covid_list.count(x)) for x in set(covid_list)]

non_list= [int(non_pred[i]) for i in range(len(non_pred))]
non_count = [(x, non_list.count(x)) for x in set(non_list)]

y_pred_list= covid_list+non_list
y_test_list= [1 for i in range(len(covid_list))]+[0 for i in range(len(non_list))]

y_pred= np.asarray(y_pred_list, dtype=np.int64)
y_test= np.asarray(y_test_list, dtype=np.int64)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)   

df_cm = pd.DataFrame(cnf_matrix, index = [i for i in class_names], columns = [i for i in class_names])

ax = sn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True, cbar=False, fmt='g', xticklabels= ['COVID olmayan','COVID'], yticklabels= ['COVID olmayan','COVID'])
ax.set_title("Hata Matrisi")
plt.savefig('C:/Users/AhmetCB/Desktop/Covid/Hata_Matrisi.png')

end_time= time.time()
tot_time= end_time- start_time
print("\nToplam Zaman:", tot_time)

