import torch 
import random
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from torch.utils.data import Dataset 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet 
import torch.nn as nn 
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader 


seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.chdir('C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7')
# # Whether to train on a gpu
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')

data_path = 'C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7/'

train_df = pd.read_csv(data_path + 'train.csv')
test_df = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

train, valid = train_test_split(train_df, test_size=0.2,stratify=train_df[['healthy', 'multiple_diseases', 'rust', 'scab']],random_state=50)

print(train_df.shape)
print(test_df.shape)


class ImageDataset(Dataset):
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__() 
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
 
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0] 
        img_path = self.img_dir + img_id + '.jpg' 
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
    
        if self.is_test:
            return image
        else:
            label = np.argmax(self.df.iloc[idx, 1:5]) 
            return image, label


img_dir = 'C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7/images/'


transform_train = A.Compose([
    A.Resize(224,224),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.VerticalFlip(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=30, p=0.3),
    A.OneOf([A.Emboss(p=1),A.Sharpen(p=1), A.Blur(p=1)], p=0.3),A.PiecewiseAffine(p=0.3),
    A.Normalize(), 
    ToTensorV2() 
])
# Test does not use augmentation
transform_test = A.Compose([ 
    A.Resize(224,224),
    A.Normalize(),
    ToTensorV2()
])

dataset_train = ImageDataset(train, img_dir=img_dir, transform=transform_train)
dataset_valid = ImageDataset(valid, img_dir=img_dir, transform=transform_test)

print(len(dataset_train))
print(len(dataset_valid))



batch_size = 4

# To avoid loading all of the data into memory at once, we use training DataLoaders. At training time, the DataLoader will load the images from disk, apply the transformations, and yield a batch. 
# To train and validation, we'll iterate through all the batches in the respective DataLoader.
# One crucial aspect is to shuffle the data before passing it to the network. This means that the ordering of the image categories changes on each pass through the data (one pass through the data is one training epoch).
trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False,num_workers=0)

train_features, train_labels = next(iter(trainloader))

# The shape of a batch is (batch_size, color_channels, height, width).
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4) 

model = model.to(device) 
print(model)

# Loss (criterion): keeps track of the loss itself and the gradients of the loss with respect to the model parameters (weights)
criterion = nn.CrossEntropyLoss()

# Optimizer: updates the parameters (weights) with the gradients
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0006, weight_decay=0.001)

epochs =10
#scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(trainloader)*3, num_training_steps=len(trainloader)*epochs)


for epoch in range(epochs):
    # set to training 
    model.train() 
    epoch_train_loss = 0 
    # Training loop
    for images, labels in tqdm(trainloader):
        # Tensors to gpu 
        images = images.to(device)
        labels = labels.to(device)
        # Clear gradients
        optimizer.zero_grad()
        # Predicted outputs are log probabilities
        outputs = model(images)
        #  Loss and backpropagation of gradients
        loss = criterion(outputs, labels)
        epoch_train_loss += loss.item() 
        loss.backward() 
        # Update the parameters
        optimizer.step()
        #scheduler.step()
        # Track train loss
    print(f' [{epoch+1}/{epochs}] -: {epoch_train_loss/len(trainloader):.4f}')
    
    # set to evaluation mode
    model.eval() 
    epoch_valid_loss = 0 
    preds_list = [] 
    true_onehot_list = []
     # Don't need to keep track of gradients
    with torch.no_grad(): 
        # Validation loop
        for images, labels in validloader:
            # Tensors to gpu
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            # Validation loss
            loss = criterion(outputs, labels)
            
            epoch_valid_loss += loss.item()
            
            preds = torch.softmax(outputs.cpu(), dim=1).numpy() 
            true_onehot = torch.eye(4)[labels].cpu().numpy() 
            preds_list.extend(preds)
            true_onehot_list.extend(true_onehot)

    print(f' [{epoch+1}/{epochs}] -: {epoch_valid_loss/len(validloader):.4f} / ROC AUC : {roc_auc_score(true_onehot_list, preds_list):.4f}') 

torch.save(model.state_dict(), "outputs/model.pth")
print("Saved PyTorch Model State to model.pth") 

dataset_test = ImageDataset(test_df, img_dir=img_dir, transform=transform_test, is_test=True)
print(len(dataset_test))
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

dataset_TTA = ImageDataset(test_df, img_dir=img_dir, transform=transform_train, is_test=True)
loader_TTA = DataLoader(dataset_TTA, batch_size=batch_size, shuffle=False, num_workers=0)

model.eval() 

preds_test = np.zeros((len(test_df), 4)) 

with torch.no_grad():
    for i, images in enumerate(loader_test):
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
        preds_test[i*batch_size:(i+1)*batch_size] += preds_part

submission_test = submission.copy()

submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_test

num_TTA = 3

preds_tta = np.zeros((len(test_df), 4)) 

for i in range(num_TTA):
    with torch.no_grad():
        for i, images in enumerate(loader_TTA):
            images = images.to(device)
            outputs = model(images)
            preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
            preds_tta[i*batch_size:(i+1)*batch_size] += preds_part


preds_tta /= num_TTA 

submission_tta = submission.copy() 

submission_tta[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_tta

submission_test.to_csv('submission_test.csv', index=False)
submission_tta.to_csv('submission_tta.csv', index=False)
