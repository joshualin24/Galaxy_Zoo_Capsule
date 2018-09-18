"""
Special Note: Small Galaxy Zoo images are written in .jpg extension
              Opencv images don't support .jpg, so they are written in .png extension

"""

import sys
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

GALAXY_IMG_FOLDER =  './v0828new/'
GALAXY_IMG_ID_FILE = './v0828overlap-id.txt'
GALAXY_TRAIN_IMG_ID_FILE = './v0828overlap-train-id.txt'
GALAXY_TEST_IMG_ID_FILE = './v0828overlap-test-id.txt'
GALAXY_IMG_LABEL_FILE = './v0828overlap-catalog.txt'

ROOT_FOLDER = './v0904/'
GALAXY_TRAIN_IMG_FOLDER = './v0904/train_img/'
GALAXY_TRAIN_LABEL_FOLDER = './v0904/train_label/'
GALAXY_TEST_IMG_FOLDER = './v0904/test_img/'
GALAXY_TEST_LABEL_FOLDER = './v0904/test_label/'

if not os.path.exists(ROOT_FOLDER):
    os.makedirs(ROOT_FOLDER)
    
if not os.path.exists(GALAXY_TRAIN_IMG_FOLDER):
    os.makedirs(GALAXY_TRAIN_IMG_FOLDER)
    
if not os.path.exists(GALAXY_TRAIN_LABEL_FOLDER):
    os.makedirs(GALAXY_TRAIN_LABEL_FOLDER)
    
if not os.path.exists(GALAXY_TEST_IMG_FOLDER):
    os.makedirs(GALAXY_TEST_IMG_FOLDER)
    
if not os.path.exists(GALAXY_TEST_LABEL_FOLDER):
    os.makedirs(GALAXY_TEST_LABEL_FOLDER)

try:
    DF_IMG_ID = pd.read_csv(GALAXY_IMG_ID_FILE, names=['ID'])
    LIST_IMG_ID = DF_IMG_ID.values.flatten()
except FileNotFoundError:
    print("IMG ID File does not found.")
    
try:
    DF_TRAIN_IMG_ID = pd.read_csv(GALAXY_TRAIN_IMG_ID_FILE, names=['ID'])
    LIST_TRAIN_IMG_ID = DF_TRAIN_IMG_ID.values.flatten()
    
    DF_TEST_IMG_ID = pd.read_csv(GALAXY_TEST_IMG_ID_FILE, names=['ID'])
    LIST_TEST_IMG_ID = DF_TEST_IMG_ID.values.flatten()
except FileNotFoundError:
    print("IMG ID File does not found.")
    
    
def train_test_split(x):
    np.random.shuffle(x)
    threshold = int(len(x)*.8)
    training, test = x[:threshold], x[threshold:]
    with open('./v0828overlap-train-id.txt','w+') as f:
        for ID in training:
            f.write(ID + '\n')
    with open('./v0828overlap-test-id.txt','w+') as f:
        for ID in test:
            f.write(ID + '\n')
     
###
#  Get the source image and label for the ith image in the train/test text file
#  Can have additional label for non-overlapping galaxies (ordered by area)
###      
def get_img_label(index, is_train=True, additional_label=1):

    if is_train:
        list_ID = LIST_TRAIN_IMG_ID
    else:
        list_ID = LIST_TEST_IMG_ID
        
    filename = GALAXY_IMG_FOLDER + list_ID[index]
    image = cv2.imread(filename,1)
    labels = get_bounding_box(image,additional_label)
    return image, labels

# create bounding box for the largest contour in the image
def get_bounding_box(image, additional_label=1,show_img=False):
    img_h, img_w = image.shape[:-1]
    img_h_2, img_w_2 = img_h//2, img_w//2

    x,y,h,w = 0,0,0,0
    label = [0,0,0,0,0]
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(im_bw, 63, 255, 0)
    _ , contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    data = [None] * len(contours)
    
    for i in range(len(contours)):
        x,y,h,w = cv2.boundingRect(contours[i])
        area = h * w
        
        if .8* img_h_2 < x + h/2 < 1.2* img_h_2 and .80* img_w_2 < y + w/2 < 1.2* img_w_2:
            data[i] = [area,1,x/img_h,y/img_w,h/img_h,w/img_w]
        else:
            data[i] = [area,0,x/img_h,y/img_w,h/img_h,w/img_w]
        
    data_overlapping = np.array([points for points in data if points[1] == 1])
    data_overlapping = data_overlapping[data_overlapping[:,0].argsort()]
    label = np.asarray([data_overlapping[-1,1:]])
             
    if show_img:
        x,y,h,w = label[1:]
        cv2.rectangle(image, (x,y), (x+h,y+w),(255,0,0),1)  
        plt.figure()
        plt.imshow(image.astype(int))
        plt.show()
    
    if additional_label >= 1:
        data_nonoverlapping = np.array([points for points in data if points[1] == 0])
        data_nonoverlapping = data_nonoverlapping[data_nonoverlapping[:,0].argsort()]
        label = np.vstack((label,data_nonoverlapping[-additional_label:,1:]))
    return label

###
#  Dihedral transformation of bbox (should be in range [0,1] for each label)
###
def flip_axis_1(x,y,h,w):
    return (1-(x+h)),y,h,w
    
def flip_axis_0(x,y,h,w):
    return x,(1-(y+w)),h,w

def rot90(x,y,h,w):
    return y,(1-(x+h)),w,h
    
def dihedral_transform(img_0,labels_0,action:0):
    img = img_0.copy()
    labels = labels_0.copy()
    
    if action == 0:
        pass
    elif action == 1:
        img = np.flip(img,axis=0)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_0(*labels[i,1:])
    elif action == 2:
        img = np.flip(img,axis=1)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_1(*labels[i,1:])
    elif action == 3:
        img = np.flip(np.flip(img,axis=0),axis=1)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_0(*labels[i,1:])
            labels[i,1:] = flip_axis_1(*labels[i,1:])
    elif action == 4:
        img = np.flip(np.flip(img,axis=1),axis=0)
        for i in range(len(labels)):
            labels[i,1:] = flip_axis_1(*labels[i,1:])
            labels[i,1:] = flip_axis_0(*labels[i,1:])
    elif action == 5:
        img = np.rot90(img,k=1)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
    elif action == 6:
        img = np.rot90(img,k=2)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
    elif action == 7:
        img = np.rot90(img,k=3)
        for i in range(len(labels)):
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
            labels[i,1:] = rot90(*labels[i,1:])
    else:
        print("Dihedral group action should be in range [0,7]")
    
    return img, labels

###
#  Corp Image By four corners (Upper Left, Upper Right, Lower Left, Lower Right)
###        
def corp_img(img, labels, corp_ratio=3./4, corner=0):
    img_h, img_w = img.shape[:-1]
    img_corp_h, img_corp_w = int(img_h*corp_ratio), int(img_w*corp_ratio)
    img_0 = img.copy()
    labels_0 = labels.copy()
    new_labels = np.asarray([])
    
    if corner == 0: # upper left
        img_0 = img_0[:img_corp_h,:img_corp_w]
        for l in labels_0:
            x,y,h,w = l[1:]
            x,y,h,w = x*img_h,y*img_w,h*img_h,w*img_w
            # Is label still valid in the corpped image?
            if x < img_corp_h and y < img_corp_w:
                if x + h > img_corp_h:
                    h = img_corp_h - x
                if y + w > img_corp_w:
                    w = img_corp_w - y
                    
                x,y,h,w = x/img_corp_h,y/img_corp_w,h/img_corp_h,w/img_corp_w
                if len(new_labels) == 0:
                    new_labels = np.append(new_labels, [l[0],x,y,h,w])
                else:
                    new_labels = np.vstack((new_labels, [l[0],x,y,h,w]))
                
    elif corner == 1: # lower right
        img_0 = img_0[img_h-img_corp_h:,img_w-img_corp_w:]
        for l in labels_0:
            x,y,h,w = l[1:]
            x,y,h,w = x*img_h,y*img_w,h*img_h,w*img_w
            # Is label still valid in the corpped image?
            if x >= img_h-img_corp_h and y >= img_w-img_corp_w:
                x -= img_h-img_corp_h
                y -= img_w-img_corp_w
                
                x,y,h,w = x/img_corp_h,y/img_corp_w,h/img_corp_h,w/img_corp_w
                if len(new_labels) == 0:
                    new_labels = np.append(new_labels, [l[0],x,y,h,w])
                else:
                    new_labels = np.vstack((new_labels, [l[0],x,y,h,w]))
                
    elif corner == 2: # upper right
        img_0 = img_0[:img_corp_w, img_w-img_corp_w:]
        for l in labels_0:
            x,y,h,w = l[1:]
            x,y,h,w = x*img_h,y*img_w,h*img_h,w*img_w
            # Is label still valid in the corpped image?
            if x >= img_h-img_corp_h and y < img_corp_w:
                x -= img_h-img_corp_h
                if y + w > img_corp_w:
                    w = img_corp_w - y
                    
                x,y,h,w = x/img_corp_h,y/img_corp_w,h/img_corp_h,w/img_corp_w
                if len(new_labels) == 0:
                    new_labels = np.append(new_labels, [l[0],x,y,h,w])
                else:
                    new_labels = np.vstack((new_labels, [l[0],x,y,h,w]))
                
    elif corner == 3: # lower left
        img_0 = img_0[img_h-img_corp_h:,:img_corp_h]
        for l in labels_0:
            x,y,h,w = l[1:]
            x,y,h,w = x*img_h,y*img_w,h*img_h,w*img_w
            # Is label still valid in the corpped image?
            if x < img_corp_h and y >= img_w-img_corp_w:
                if x + h > img_corp_h:
                    h = img_corp_h - x
                y -= img_w-img_corp_w
                
                x,y,h,w = x/img_corp_h,y/img_corp_w,h/img_corp_h,w/img_corp_w
                if len(new_labels) == 0:
                    new_labels = np.append(new_labels, [l[0],x,y,h,w])
                else:
                    new_labels = np.vstack((new_labels, [l[0],x,y,h,w]))
    else:
        new_labels = labels
    
    # if only one label, expand its batch size dim
    if len(new_labels) == 5:
        new_labels = np.expand_dims(new_labels,axis=0)
        
    return img_0, new_labels

###
#  Main Function to generate training/testing dataset
###
def generate_train_img_label(is_train=True,additional_label=0):

    if is_train:
        iters = len(LIST_TRAIN_IMG_ID)
        img_folder = GALAXY_TRAIN_IMG_FOLDER
        label_folder = GALAXY_TRAIN_LABEL_FOLDER
        catalog = 'train_catalog.txt'
    else:
        iters = len(LIST_TEST_IMG_ID)
        img_folder = GALAXY_TEST_IMG_FOLDER
        label_folder = GALAXY_TEST_LABEL_FOLDER
        catalog = 'test_catalog.txt'
    
    with open(ROOT_FOLDER + catalog, 'w+') as cat:
        for i in tqdm(range(iters)):
            img,labels = get_img_label(i, is_train, additional_label)
            img_id = LIST_TRAIN_IMG_ID[i][:-5]
            for action in range(8):
                img_0, labels_0 = dihedral_transform(img,labels,action=action)
                for corner in range(5):
                    img_1, labels_1 = corp_img(img_0, labels_0, corp_ratio=3./4, corner=corner)
                    if len(labels_1) >= 1:
                        img_name = '{}_action{}_corner{}'.format(img_id,action,corner)
                        # write image
                        cv2.imwrite(img_folder + img_name + '.png', img_1)
                        # write img name to catalog
                        cat.write(img_name+'\n')
                        # write label to text file
                        with open(label_folder + img_name + '.txt', 'w+') as f:
                            for l in labels_1:
                                f.write('{} {} {} {} {}\n'.format(*l))

if __name__=='__main__':
    
    generate_train_img_label(is_train=False,additional_label=3)
    
    # funtionality test
    """
    img, labels = get_img_label(index=0, is_train=True, additional_label=2)
    print(labels)
    colors = {0:(255,0,0),1:(0,255,0)}
    img_0 = img.copy()
    for label in labels:
        bb0 = label[1:].copy()
        bb0[0] *= img_0.shape[0]
        bb0[1] *= img_0.shape[1]
        bb0[2] *= img_0.shape[0]
        bb0[3] *= img_0.shape[1]
        bb0 = [int(e) for e in bb0]
        cv2.rectangle(img_0, (bb0[0],bb0[1]), (bb0[0]+bb0[2],bb0[1]+bb0[3]),colors[0], 0)
        
    plt.figure()
    plt.imshow(img_0.astype(int))
    plt.show()
    
    for corner in range(4):
        img_0, labels_0 = corp_img(img, labels, corp_ratio=3./4, corner=corner)
        print(labels_0)
        img_0 = img_0.copy()
        for label in labels_0:
            bb0 = label[1:].copy()
            bb0[0] *= img_0.shape[0]
            bb0[1] *= img_0.shape[1]
            bb0[2] *= img_0.shape[0]
            bb0[3] *= img_0.shape[1]
            bb0 = [int(e) for e in bb0]
            cv2.rectangle(img_0, (bb0[0],bb0[1]), (bb0[0]+bb0[2],bb0[1]+bb0[3]),colors[0], 0)
            
        plt.figure()
        plt.imshow(img_0.astype(int))
        plt.show()
    
    for action in range(8):
        img_0, labels_0 = dihedral_transform(img,labels,action=action)
        print(labels_0)
        img_0 = img_0.copy()
        for label in labels_0:
            bb0 = label[1:].copy()
            bb0[0] *= img_0.shape[0]
            bb0[1] *= img_0.shape[1]
            bb0[2] *= img_0.shape[0]
            bb0[3] *= img_0.shape[1]
            bb0 = [int(e) for e in bb0]
            cv2.rectangle(img_0, (bb0[0],bb0[1]), (bb0[0]+bb0[2],bb0[1]+bb0[3]),colors[0], 0)
            
        plt.figure()
        plt.imshow(img_0.astype(int))
        plt.show()
    """
