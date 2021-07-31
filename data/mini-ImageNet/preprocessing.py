# Transfer the mini-imagenet from meta-learning to classification
# link: https://github.com/yaoyao-liu/mini-imagenet-tools/tree/main/csv_files


import os
import pandas as pd
from PIL import Image

trainval_data = pd.read_csv("/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/trainval.csv")
test_data = pd.read_csv("/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/test.csv")
frames = [trainval_data, test_data]
all_data = pd.concat(frames)

def generate_data(all_data):
    images = []
    labels = []
    for item in all_data.values:
        name = item[0]
        label = item[1]
        image = 'train/' + label + '/' + name
        images.append(image)
        labels.append(label)
    return images, labels

def generate_split(images, labels):
    category = list(set(labels))
    data_dict = {}
    for image, label in zip(images, labels):
        if label in data_dict:
            data_dict[label].append(image)
        else:
            data_dict[label] = [image,]
    train_split = []
    test_split = []
    val_split = []
    for label in category:
        train = data_dict[label][:420]
        test = data_dict[label][420:540]
        val = data_dict[label][540:]
        for item in train:
            train_split.append([item, label])
        for item in test:
            test_split.append([item, label])
        for item in val:
            val_split.append([item, label])
    return train_split, test_split, val_split, category


images, labels = generate_data(all_data)
train_split, test_split, val_split, category = generate_split(images, labels)

train_frame = {'image': [item[0] for item in train_split], 
               'label': [item[1] for item in train_split],}
test_frame = {'image': [item[0] for item in test_split], 
              'label': [item[1] for item in test_split],}
val_frame = {'image': [item[0] for item in val_split], 
             'label': [item[1] for item in val_split],}
category_frame = {'category': category,}
     
category_frame = pd.DataFrame(data=category_frame)
train_frame = pd.DataFrame(data=train_frame)
test_frame = pd.DataFrame(data=test_frame)
val_frame = pd.DataFrame(data=val_frame)

category_frame.to_csv('/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/category_frame.csv')
train_frame.to_csv('/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/train_frame.csv')
test_frame.to_csv('/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/test_frame.csv')
val_frame.to_csv('/home/username/projects/Causal_Adversarial.private/BPFC_mine/data/mini-ImageNet/val_frame.csv')
