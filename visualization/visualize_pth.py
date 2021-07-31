import math
import torch
import numpy as np
from PIL import Image

saved_data = torch.load('/home/username/checkpoints/cifar10_decouple_lv_hv/adv_images.pth')

index = 10

# show org image
img = (saved_data['org_images'][index]*255).permute(1,2,0).numpy().astype('uint8')
Image.fromarray(img).resize((128, 128))

# show adv image
adv = (saved_data['adv_images'][index]*255).permute(1,2,0).numpy().astype('uint8')
Image.fromarray(adv).resize((128, 128))

def get_bit_image(image, bit=1):
    p_val = math.pow(2, bit)
    image = torch.round(image * 255)
    image = torch.remainder(image, p_val * 2)
    image = (image - torch.remainder(image, p_val))
    image = image / 255
    return image

def show_img(image):
    image = image * 255
    image = image.permute(1,2,0).numpy().astype('uint8')
    return Image.fromarray(image).resize((128, 128))

def multi_bit_image(image, bits):
    images = [get_bit_image(image, i) for i in bits]
    return sum(images)

def differentiable_remainder(x, divider):
    return x - (torch.floor_divide(x, divider) * divider).clone().detach()

index1 = 10
index2 = 20

org_image1 = saved_data['org_images'][index1]
adv_image1 = saved_data['adv_images'][index1]
org_image2 = saved_data['org_images'][index2]
adv_image2 = saved_data['adv_images'][index2]

show_img(org_image1)

show_img(org_image2)

show_img(multi_bit_image(org_image2, [1,2,3,4,5]) + multi_bit_image(org_image1, [6,7]))
