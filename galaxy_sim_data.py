"""
Special Note: Small Galaxy Zoo images are written in .jpg extension
              Opencv images don't support .jpg, so they are written in .png extension

"""

import numpy as np
import pandas as pd
import cv2
import sys

GALAXY_TRAIN_FILE = './data/train_simple.txt'
GALAXY_TEST_FILE = './data/test_simple.txt'
GALAXY_ORIG_FOLDER = './data/images_training_rev1/'
GALAXY_TRAIN_FOLDER = './data/train_samples/'

DF_TRAIN = pd.read_csv(GALAXY_TRAIN_FILE,names=['ID','label'])
DF_TEST = pd.read_csv(GALAXY_TEST_FILE,names=['ID','label'])

OFF_SET = 25

# get data batch
def get_img_label(df_samples, batch_size):
    images = np.zeros([batch_size, 212, 212, 1])
    labels = np.zeros([batch_size,2])
    images_id = list(df_samples['ID'])
    images_label = list(df_samples['label'])
    for i in range(batch_size):
        filename = GALAXY_ORIG_FOLDER + '%d' % images_id[i] + '.jpg'
        images[i] = np.asarray(cv2.resize(cv2.imread(filename,0),(212,212))).reshape((212,212,1))
        if images_label[i] == 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return images, labels

def galaxy_train_next_batch(batch_size, get_image_label_func=get_img_label):
    df_samples = DF_TRAIN.sample(batch_size)
    return get_image_label_func(df_samples, batch_size)

def galaxy_test_next_batch(batch_size,  get_image_label_func=get_img_label):
    df_samples = DF_TEST.sample(batch_size)
    return get_image_label_func(df_samples, batch_size)

# argument data with shift offsets
def augmentation(x,max_offset=2):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:]

# training & testing generators
def galaxy_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch = galaxy_train_next_batch(batch_size)
        images = batch[0]
        images = np.concatenate([images]*3,axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def galaxy_test_iter(iters=1000,batch_size=32,is_shift_ag=False):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch = galaxy_test_next_batch(batch_size)
        images = batch[0]
        images = np.concatenate([images] * 3, axis=-1)
        yield augmentation(images,max_offset), np.stack([batch[1]]*3, axis=-1)

def multigalaxy_train_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch1 = galaxy_train_next_batch(batch_size)
        batch2 = galaxy_train_next_batch(batch_size)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        images = np.clip(np.add(images1,images2).astype(np.float32),0,255)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)

def multigalaxy_test_iter(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        batch1 = galaxy_test_next_batch(batch_size)
        batch2 = galaxy_test_next_batch(batch_size)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        images = np.clip(np.add(images1,images2).astype(np.float32),0,255)
        images = np.concatenate([images,images1,images2], axis=-1)
        y1,y2 = batch1[1],batch2[1]
        y0 = np.logical_or(y1,y2).astype(np.float32)
        yield images, np.stack([y0,y1,y2], axis=-1)


# Code for 224x224x3 input size

GALAXY_TRAIN_FOLDER =  './data/train_samples/'
GALAXY_TRAIN_FILE = './data/train_samples/lens_parameters.txt'
try:
    DF_LABEL = pd.read_csv(GALAXY_TRAIN_FILE, names=['eplliptical','spiral'])
    LIST_LABEL = DF_LABEL.values
except FileNotFoundError:
    print("Training dataset has not created yet!")
   
# get data batch from training/testing dataset
def get_img_label_alexnetV2(start_index, batch_size):
    images = np.zeros([batch_size, 224, 224, 3])
    start_index = start_index * batch_size
    err = False
    for i in range(batch_size):
        filename = GALAXY_TRAIN_FOLDER + "img" + '_' + "%07d" % (start_index+i+1) + '.png'
        img = cv2.imread(filename,0)
        if img is not None:
            images[i] = np.asarray(cv2.resize(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),(224,224))).reshape((224,224,3))
        else:
            err = True
    labels = LIST_LABEL[start_index:start_index+batch_size]
    return images, labels, err

def multigalaxy_train_iter_alexnet(iters=1000,batch_size=32,is_shift_ag=True):
    max_offset = int(is_shift_ag) * OFF_SET
    for i in range(iters):
        yield get_img_label_alexnetV2(i, batch_size)

# get data batch from smallGalaxyZoo
def get_img_label_alexnet(df_samples, batch_size):
    images = np.zeros([batch_size, 224, 224, 3])
    labels = np.zeros([batch_size,2])
    images_id = list(df_samples['ID'])
    images_label = list(df_samples['label'])
    for i in range(batch_size):
        filename = GALAXY_ORIG_FOLDER + '%d' % images_id[i] + '.jpg'
        images[i] = np.asarray(cv2.resize(cv2.cvtColor(cv2.imread(filename,0),cv2.COLOR_GRAY2RGB),(224,224))).reshape((224,224,3))
        if images_label[i] == 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return images, labels

# genreate versatile categories of data
def multigalaxy_generate_sample_alexnet(iters=1000,batch_size=1,is_shift_ag=True, is_train = True, noFN=True):
    max_offset = int(is_shift_ag) * OFF_SET
    coins = np.random.uniform(size=iters)
    if is_train:
        get_batch_func = galaxy_train_next_batch
    else:
        get_batch_func = galaxy_test_next_batch
    for i in range(iters):
        batch1 = get_batch_func(batch_size, get_img_label_alexnet)
        batch2 = get_batch_func(batch_size, get_img_label_alexnet)
        images1 = augmentation(batch1[0],max_offset)
        images2 = augmentation(batch2[0],max_offset)
        y1,y2 = batch1[1],batch2[1]
        # noFN means not including any False-Negative samples
        if not noFN:
            if coins[i] < 0.25:
                images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
                y0 = np.logical_or(y1,y2).astype(np.float32)
            elif 0.25 <= coins[i] < 0.5:
                images = images1
                y0 = y1
            elif 0.5 <= coins[i] < 0.75:
                images = images2
                y0 = y2
            else:
                images = np.asarray([255 * np.random.random((224,224,3))])
                y0 = np.zeros([1,2])
        else:
            if coins[i] <= 0.33:
                images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
                y0 = np.logical_or(y1,y2).astype(np.float32)
            elif 0.33 < coins[i] <= 0.66:
                images = images1
                y0 = y1
            else:
                images = images2
                y0 = y2
            
        yield images, y0


def get_img_labelV2(df_samples, batch_size):
    images = np.zeros([batch_size, 224, 224, 3])
    labels = np.zeros([batch_size,6])
    images_id = list(df_samples['ID'])
    images_label = list(df_samples['label'])
    for i in range(batch_size):
        filename = GALAXY_ORIG_FOLDER + '%d' % images_id[i] + '.jpg'
        image = cv2.resize(cv2.imread(filename,0),(224,224))
        images[i] = np.array(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)).reshape((224,224,3))
        if images_label[i] == 0:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
        labels[i][2:] = get_bounding_box(image)
    return images, labels

# create bounding box for the largest contour in the image
def get_bounding_box(image):
    max_area = -1
    x,y,h,w = 0,0,0,0
    label = [0,0,0,0]
    ret, thresh = cv2.threshold(image, 63, 255, 0)
    _ , contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x,y,h,w = cv2.boundingRect(contours[i])
        area = h * w
        # making sure to select the largest galaxy in the center of the image
        if max_area < area and 90 < x + h/2 < 130 and 90 < y + w/2 < 130:
            max_area = area
            label = [x,y,h,w]
    assert(max_area != -1)
    return label

def augmentationV2(x,max_offset=50):
    bz,h,w,c = x.shape
    bg = np.zeros([bz,w+2*max_offset,h+2*max_offset,c])
    offsets = np.random.randint(0,2*max_offset+1,2)
    bg[:,offsets[0]:offsets[0]+h,offsets[1]:offsets[1]+w,:] = x
    return bg[:,max_offset:max_offset+h,max_offset:max_offset+w,:], offsets
    
def multigalaxy_generate_sample_alexnetV2(iters=1000,batch_size=1,is_shift_ag=True, is_train = True, noFN=True):
    max_offset = int(is_shift_ag) * OFF_SET
    coins = np.random.uniform(size=iters)
    if is_train:
        get_batch_func = galaxy_train_next_batch
    else:
        get_batch_func = galaxy_test_next_batch
    for i in range(iters):
        batch1 = get_batch_func(batch_size, get_img_labelV2)
        batch2 = get_batch_func(batch_size, get_img_labelV2)
        images1, offset1 = augmentationV2(batch1[0],max_offset)
        images2, offset2 = augmentationV2(batch2[0],max_offset)
        y1,y2 = batch1[1][:,:2],batch2[1][:,:2]
        bb1,bb2 = batch1[1][:,2:],batch2[1][:,2:]
        #print(bb1,offset1,bb2,offset2)
        bb1[:,0], bb1[:,1] = bb1[:,0]+offset1[0]-max_offset, bb1[:,1]+offset1[1]-max_offset
        bb2[:,0], bb2[:,1] = bb2[:,0]+offset2[0]-max_offset, bb2[:,1]+offset2[1]-max_offset
        # noFN means not including any False-Negative samples
        if not noFN:
            if coins[i] < 0.25:
                images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
                y0 = np.logical_or(y1,y2).astype(np.float32)
            elif 0.25 <= coins[i] < 0.5:
                images = images1
                y0 = y1
                bb2 = bb1
            elif 0.5 <= coins[i] < 0.75:
                images = images2
                y0 = y2
                bb1 = bb2
            else:
                images = np.asarray([255 * np.random.random((224,224,3))])
                y0 = np.zeros([1,2])
        else:
            if coins[i] <= 0.33:
                images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
                y0 = np.logical_or(y1,y2).astype(np.float32)
            elif 0.33 < coins[i] <= 0.66:
                images = images1
                y0 = y1
                bb2 = bb1
            else:
                images = images2
                y0 = y2
                bb1 = bb2
            
        yield images, y0, bb1, y1, bb2,y2
        
def multigalaxy_generate_sample_alexnetV3(iters=1000,batch_size=1,is_shift_ag=True, is_train = True, noFN=True):
    max_offset = int(is_shift_ag) * OFF_SET
    coins = np.random.uniform(size=iters)
    if is_train:
        get_batch_func = galaxy_train_next_batch
    else:
        get_batch_func = galaxy_test_next_batch
    for i in range(iters):
        batch1 = get_batch_func(batch_size, get_img_labelV2)
        batch2 = get_batch_func(batch_size, get_img_labelV2)
        
        images1, offset1 = augmentationV2(batch1[0],max_offset)
        images2, offset2 = augmentationV2(batch2[0],max_offset)
        y1,y2 = batch1[1][:,:2],batch2[1][:,:2]
        bb1,bb2 = batch1[1][:,2:],batch2[1][:,2:]
        #print(bb1,offset1,bb2,offset2)
        bb1[:,0], bb1[:,1] = bb1[:,0]+offset1[0]-max_offset, bb1[:,1]+offset1[1]-max_offset
        bb2[:,0], bb2[:,1] = bb2[:,0]+offset2[0]-max_offset, bb2[:,1]+offset2[1]-max_offset
        # noFN means not including any False-Negative samples
        images = np.clip(np.add(images1,images2),0,255).astype(np.float32)
        y0 = np.logical_or(y1,y2).astype(np.float32)
        bb0 = np.zeros_like(bb1)
        bb0[:,0] = min(bb1[:,0],bb2[:,0])
        bb0[:,1] = min(bb1[:,1],bb2[:,1])
        bb0[:,2] = max(bb1[:,0]+bb1[:,2],bb2[:,0]+bb2[:,2]) - bb0[:,0]
        bb0[:,3] = max(bb1[:,1]+bb1[:,3],bb2[:,1]+bb2[:,3]) - bb0[:,1]
        
        batch3 = get_batch_func(batch_size, get_img_labelV2)
        images3 = batch3[0]
        
        images0 = np.zeros_like(images)
        size = images0.shape[1]
        
        resize2 = np.random.randint(low=size//2,high=size)
        x2,y2 = np.random.randint(size-resize2), np.random.randint(size-resize2)
        images3 = cv2.resize(images3[0],(resize2,resize2))
        
        resize1 = np.random.randint(low=size//5,high=size//2)
        x1,y1 = np.random.randint(size-resize1), np.random.randint(size-resize1)
        images = cv2.resize(images[0],(resize1,resize1))
        
        
        
        images0[:,x2:x2+resize2,y2:y2+resize2] += images3
        images0[:,x1:x1+resize1,y1:y1+resize1] += images
        
        bb0 = bb0 * (resize1 / size)
        bb0[:,0] += x1
        bb0[:,1] += y1 
            
        yield images0, bb0, y0, bb1, y1, bb2,y2


if __name__=='__main__':
    # funtionality test
    
    import matplotlib.pyplot as plt
    train = multigalaxy_generate_sample_alexnetV3(iters=20, batch_size=1)
    colors = {0:(255,0,0),1:(0,255,0)}
    green = (0,255,0)
    for img, bb0, y0, bb1, y1, bb2,y2 in train:
        print(y0, bb0, y1, bb1, y2, bb2)
        img_0 = img[0].copy()
        bb0 = [int(e) for e in bb0[0]]
        
        cv2.rectangle(img_0, (bb0[1],bb0[0]), (bb0[1]+bb0[3],bb0[0]+bb0[2]),colors[0], 1)
            
        plt.figure()
        plt.imshow(img_0.astype(int))
        plt.show()
    
    """
    train = multigalaxy_generate_sample_alexnetV2(iters=20, batch_size=1)
    colors = {0:(255,0,0),1:(0,255,0)}
    green = (0,255,0)
    for img,y0, bb1, y1, bb2,y2 in train:
        print(y0, y1, bb1, y2, bb2)
        img_0 = img[0].copy()
        bb1 = [int(e) for e in bb1[0]]
        bb2 = [int(e) for e in bb2[0]]
        
        if np.array_equal(y0[0],y1[0]):
            cv2.rectangle(img_0, (bb1[1],bb1[0]), (bb1[1]+bb1[3],bb1[0]+bb1[2]),colors[y1[0,1]], 1)
        elif np.array_equal(y0[0],y2[0]):
            cv2.rectangle(img_0, (bb2[1],bb2[0]), (bb2[1]+bb2[3],bb2[0]+bb2[2]),colors[y2[0,1]], 1)
        else:
            cv2.rectangle(img_0, (bb1[1],bb1[0]), (bb1[1]+bb1[3],bb1[0]+bb1[2]),colors[y1[0,1]], 1)
            cv2.rectangle(img_0, (bb2[1],bb2[0]), (bb2[1]+bb2[3],bb2[0]+bb2[2]),colors[y2[0,1]], 1)
            
        plt.figure()
        plt.imshow(img_0.astype(int))
        plt.show()
    """
