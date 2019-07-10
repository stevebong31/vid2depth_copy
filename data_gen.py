import os, glob, cv2, math, csv, tqdm, random
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def read_image(image_path):
    return cv2.resize(np.array(Image.open(image_path)), (416,128))/255.
        
def augment_image_colorspace(image):
    # input shape [h, w * seq , c]
    random_gamma = random.uniform(0.8,1.2)
    random_brightness = random.uniform(0.5, 2.0)
    random_colors = [random.uniform(0.8,1.2), random.uniform(0.8,1.2), random.uniform(0.8,1.2)]
    # Randomly shift gamma.
    image_aug = image**random_gamma
    # Randomly shift brightness. 
    image_aug *= random_brightness
    # Randomly shift color. 
    white = np.ones([image.shape[0], image.shape[1]])
    color_image = np.stack([white * random_colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    # Saturate.
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug

def pack_images_width(img1, img2, img3):
    return np.concatenate((img1,img2,img3), axis=1)

def unpack_images_stack(image_seq):
    width = np.int(image_seq.shape[1]/3)
    img1_f = np.concatenate((image_seq[:,:width,:],image_seq[:,width:width *2,:]), axis = 2)
    img2_f = np.concatenate((image_seq[:,width:width *2,:],image_seq[:,width *2:,:]), axis = 2)
    img1_b = np.concatenate((image_seq[:,width *2:,:],image_seq[:,width:width *2,:]), axis = 2)
    img2_b = np.concatenate((image_seq[:,width:width *2,:],image_seq[:,:width,:]), axis = 2)
    return img1_f, img2_f, img1_b, img2_b

def dataset_list_loader(kitti_path):
    dir_lists = sorted(os.listdir(kitti_path))
    total = []
    for i in dir_lists:
        paths = sorted(glob.glob(kitti_path + i + '/image_02/data/*.png'))
        x = [[paths[i], paths[i+1], paths[i+2]] for i in range(len(paths)-2)]
        for j in x:
            total.append(j)          
    return total

def img_aug_total(img_path):
    img_concat = pack_images_width(read_image(img_path[0]), read_image(img_path[1]), read_image(img_path[2]))
    return unpack_images_stack(augment_image_colorspace(img_concat))

def data_generator(total_img, batch):
    total = []
    random.shuffle(total_img)
    idx = 0
    while 1:   
        bat_img = []
        if idx > len(total_img) - batch:
            tmp_path = total_img[idx:]
            idx = 0
        else:
            tmp_path = total_img[idx:idx+16]
            idx = idx + batch

        for i in tmp_path:
            img1, img2, img3, img4 = img_aug_total(i)
            bat_img.append(img1)
            bat_img.append(img2)
            bat_img.append(img3)
            bat_img.append(img4)
        
        yield np.array(bat_img)