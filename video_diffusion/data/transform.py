import random

import torch


def short_size_scale(images, size):
    h, w = images.shape[-2:]
    short, long = (h, w) if h < w else (w, h)

    scale = size / short
    long_target = int(scale * long)

    target_size = (size, long_target) if h < w else (long_target, size)

    return torch.nn.functional.interpolate(
        input=images, size=target_size, mode="bilinear", antialias=True
    )



def random_short_side_scale(images, size_min, size_max):
    size = random.randint(size_min, size_max)
    return short_size_scale(images, size)

def short_size_scale2(images, size):
    h, w = images.shape[-2:]
    short, long = (h, w) if h < w else (w, h)

    scale = size / short
    long_target = int(scale * long)

    target_size = (size, size) 
    for i in range(20):
        # 计算需要补零的像素数
        pad_h = max(0, target_size[0] - long)
        pad_w = max(0, target_size[1] - short)
        pad_l = max(0, (target_size[1]-short)//2)
        pad_r = max(0, target_size[1]-(target_size[1]-short)//2)

        # 创建一个空白的tensor，用于填充
        padded_images = torch.zeros((3,images.shape[1],target_size[0], target_size[1]))
        # 将原始图片按按照比例分割成四个部分，并填充到空白tensor中
    
        x1 = int(short * 0)
        y1 = int(long*0)
        x2 = int(short * (0 + 1))
        y2 = int(long)
        # print("x1: ",x1,"x2: ",x2,"y1: ",y1,"y2: ",y2,"pad_h: ",pad_h,"pad_w: ",pad_w,"pad_l: ",pad_l,"pad_r: ",pad_r)
        # x1:  0 x2:  384 y1:  384 y2:  1066 pad_h:  128 pad_w:  128 pad_l:  0 pad_r:  0
        # 将当前部分复制到空白tensor中，并用零填充缺失的像素
        padded_images[:,i,y1:y2, x1+pad_w:x2+pad_w] = images[:,i,:,:]

    return padded_images

def random_crop(images, height, width):
    image_h, image_w = images.shape[-2:]
    h_start = random.randint(0, image_h - height)
    w_start = random.randint(0, image_w - width)
    return images[:, :, h_start : h_start + height, w_start : w_start + width]


def center_crop(images, height, width):
    # offset_crop(images, 0,0, 200, 0)
    image_h, image_w = images.shape[-2:]
    h_start = (image_h - height) // 4
    w_start = (image_w - width) // 2
    return images[:, :, 0 : height, w_start : w_start + width]

# # #add modification 0304
# def random_crop(images, height, width):
#     return images
# def center_crop(images, height, width):
#     return images

def offset_crop(image, left=0, right=0, top=200, bottom=0):

    n, c, h, w = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - 1)
    bottom = min(bottom, h - top - 1)
    image = image[:, :, top:h-bottom, left:w-right]

    return image