import json
import torch
import torch.nn.functional as F
import h5py
import os
import xml




######################################
#     Generate Validation data
######################################


datalist = os.listdir('/data4/imagenet/ILSVRC/Annotations/CLS-LOC/val/')

name_to_label = {}
for item in datalist:
    if item.split('.')[1] != 'xml':
        continue
    file_path = os.path.join('/data4/imagenet/ILSVRC/Annotations/CLS-LOC/val/', item)
    file = xml.dom.minidom.parse(file_path)
    labels = []
    for obj in file.getElementsByTagName('object'):
        assert len(obj.getElementsByTagName('name')) == 1
        labels.append(obj.getElementsByTagName('name')[0].firstChild.data)
    label = list(set(labels))
    assert len(label) == 1
    name = item.split('.')[0]
    name_to_label[name] = label[0]

with open('./data/ImageNet/val_info.json', 'w') as f:
    json.dump(name_to_label, f)