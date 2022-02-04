import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2





PATH = 'C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7/'
IMAGE_PATH = os.path.join(PATH,"train_img/")

train = pd.read_csv('C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('C:/ritika/kaggle_hackathons/Plant Pathology 2020 - FGVC7/plant-pathology-2020-fgvc7/test.csv')

heading = ['No. of training images', 'No. of test images']
vals = [len(train), len(test)]
plt.bar(heading, vals, color=['red', 'blue'])
for idx, val in enumerate(vals):
    plt.text(idx, val, str(val))
plt.title("No. of Images")
plt.show()


heading = train.columns[1:]
vals = [ sum(train[col] > 0) for col in heading]
plt.bar(heading, vals, color=['red', 'blue', 'green', 'orange'])
for idx, val in enumerate(vals):
    plt.text(idx, val, str(val))
plt.title("No. of Images in each Label")
plt.show()

def random_image(data, label):
    c = np.random.choice(data[data[label]>0]['image_id'])
    path = IMAGE_PATH + c + '.jpg'
    return cv2.imread(path)

fig, axarr = plt.subplots(2,2, figsize=(10,8))
axarr[0,0].imshow(random_image(train, heading[0]))
axarr[0,0].set_title(heading[0])
axarr[0,1].imshow(random_image(train, heading[1]))
axarr[0,1].set_title(heading[1])
axarr[1,0].imshow(random_image(train, heading[2]))
axarr[1,0].set_title(heading[2])
axarr[1,1].imshow(random_image(train, heading[3]))
axarr[1,1].set_title(heading[3])
fig.suptitle("Example Images", fontsize=16)
plt.show()


def plot_colour_hist(df, title):
    red_values = []; green_values = []; blue_values = []; all_channels = []
    for _, row in df.iterrows():
        img = np.array(Image.open(IMAGE_PATH + row.image_id + '.jpg'))
        red_values.append(np.mean(img[:, :, 2]))
        green_values.append(np.mean(img[:, :, 1]))
        blue_values.append(np.mean(img[:, :, 0]))
        all_channels.append(np.mean(img))
        
    _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(16, 3), sharey=True)
    for ax, column, vals, c in zip(
        axes,
        ['red', 'green', 'blue', 'all colours'],
        [red_values, green_values, blue_values, all_channels],
        'rgbk'
    ):
        ax.hist(vals, bins=100, color=c)
        ax.set_title(f'{column} hist')

    plt.suptitle(title)
    plt.show()

plot_colour_hist(train, title='Train colour dist')

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, learning_rate=100)
num_examples = 200

X = [cv2.imread(IMAGE_PATH + c + '.jpg').flatten() for c in train['image_id'][:num_examples]]
target_ids = np.where(np.array(train[heading][:num_examples]))[1]

X_tsne = tsne.fit_transform(X)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c']
for i, c, label in zip(target_ids, colors, heading):
    plt.scatter(X_tsne[target_ids == i, 0], X_tsne[target_ids == i, 1], c=c, label=label)
plt.legend()
plt.show()
