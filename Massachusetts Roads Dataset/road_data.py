
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TransfoXLCorpus
import helper
import torch
import torch.optim  as optim
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
from  torchsummary import summary
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss,JaccardLoss
from sklearn.model_selection import train_test_split

os.chdir('C:/ritika/kaggle_hackathons')

CSV_FILE='road_dataset/train.csv'
DATA_DIR='road_dataset/'

DEVICE='cpu'
EPOCHS=2
LR=0.003
BATCH_SIZE=8
IMG_SIZE=512

ENCODER= 'timm-efficientnet-b0'
WEIGHTS='imagenet'


sample_img= cv2.imread('road_dataset/images/10528675_15.png')
print(sample_img.shape)
df=pd.read_csv(CSV_FILE)

print(df.head())

#idx=20
#row=df.iloc[idx]

#image_path= DATA_DIR + row.images
#mask_path= DATA_DIR + row.masks
# reading the sample image
#image=cv2.imread(image_path)
#image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

##mask= cv2.imread(mask_path)
#mask= cv2.cvtColor(mask,cv2.IMREAD_GRAYSCALE)/255

#f,(ax1,ax2)= plt.subplots(1,2, figsize=(10,5))

#ax1.set_title('IMAGE')
#ax1.imshow(image)

#ax2.set_title('GROUND TRUTH')
#ax2.imshow(mask,cmap='gray')

#plt.show()


train_df,valid_df= train_test_split(df,test_size=0.20,random_state=42)
print(len(train_df))
print(len(valid_df))


# Augmentation transforms
def train_augs():
    return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5)])

def valid_augs():
    return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE)])

# create custom dataset
class SegmentationDataset(Dataset):

    def __init__(self,df,augmentations):
        self.df=df
        self.augmentations=augmentations
    

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        image_path= DATA_DIR + row.images
        mask_path= DATA_DIR + row.masks
        image=cv2.imread(image_path)
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask= cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask= np.expand_dims(mask,axis=-1)

        if self.augmentations:
            data= self.augmentations(image=image,mask=mask)
            image= data['image']
            mask= data['mask']

        image= np.transpose(image,(2,0,1)).astype(np.float32)
        mask= np.transpose(mask,(2,0,1)).astype(np.float32)
        image= torch.Tensor(image) / 255.0
        mask= torch.round(torch.Tensor(mask) /255.0)

        return image,mask

trainset= SegmentationDataset(train_df,train_augs())
validset= SegmentationDataset(valid_df,valid_augs())

print(len(trainset))
print(len(validset))

idx=25

image,mask= trainset[idx]
helper.show_image(image,mask)
plt.show()

# Load the dataset into batches

train_loader= DataLoader(trainset,batch_size=BATCH_SIZE,num_workers=0,shuffle=True)
valid_loader= DataLoader(validset,batch_size=BATCH_SIZE)
## total no of batches in trainloader ####

print(len(train_loader))

print(len(valid_loader))

for i,m in train_loader:
    print(f"One batch image shape :{i.shape }")
    print(f"One batch mask shape :{m.shape}")
    break;



##### Creating Segmentation model ####################################
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel,self).__init__()
        self.backbone= smp.Unet(encoder_name=ENCODER, encoder_weights=WEIGHTS, in_channels=3,classes=1,activation=None)


    def forward(self,images,masks=None):
        logits= self.backbone(images)
        if mask != None :
            return logits,DiceLoss(mode='binary')(logits,masks) + nn.BCEWithLogitsLoss()(logits,masks)
        else:
            return logits

model = SegmentationModel()
print(model)
model.to(DEVICE)

###### Creating train and valid function #####

def train_fn(dataloader,model,optimizer):
    ### set the model to training mode
    model.train()
    total_loss=0.0
    for images,masks in tqdm(dataloader):
        images=images.to(DEVICE)
        masks= masks.to(DEVICE)
        optimizer.zero_grad()
        logits,loss =model(images,masks)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    return total_loss/len(dataloader)


def valid_fn(dataloader,model):
    # set the model to evaluation mode
    model.eval()
    total_loss=0.0
    with torch.no_grad():
        for images,masks in tqdm(dataloader):
            images=  images.to(DEVICE)
            masks= masks.to(DEVICE)
            logits,loss= model(images,masks)
            total_loss+=loss.item()
        return total_loss/len(dataloader)


optimizer= torch.optim.Adam(model.parameters(),lr=LR)
best_loss= np.Inf

for i in range(EPOCHS):
    train_loss= train_fn(train_loader,model,optimizer)
    valid_loss= valid_fn(valid_loader,model)

    if valid_loss < best_loss:
        torch.save(model.state_dict(),'road_dataset/best_model.pth')
        print("saved model")
        best_loss= valid_loss
    
    print(f"Epoch:{i+1} Train Loss:{train_loss} Valid_loss :{valid_loss}")


idx=30
model.load_state_dict(torch.load('road_dataset/best_model.pth'))
image,mask= validset[idx]
# adding extra dimension for the batch
logits_mask= model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask= (pred_mask>0.5) *1.0

helper.show_image(image,mask,pred_mask.detach().cpu().squeeze(0))
plt.show()



