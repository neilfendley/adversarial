import os
from keras.utils.np_utils import to_categorical
import cv2
from keras.preprocessing import image
from keras.preprocessing import image
import json 
import params
import numpy as np
import random


def prep_image(filepath, train_dir):
    filename = os.path.basename(filepath)
    jsonData = json.load(open(filepath))
    img_filename = filepath[:-5] + '.jpg' 
    basename = os.path.basename(img_filename[:-4])
    allResults = []
    allCats = []
    for bb in jsonData['bounding_boxes']:
        category = bb['category']
        box = bb['box']

        outBaseName = basename+'_'+ ('%d' % bb['ID']) + '.jpg'

        cat_dir = os.path.join(train_dir, category+'/')
        currOut = os.path.join(cat_dir, outBaseName)
        if not os.path.exists(currOut):
            if not os.path.exists(cat_dir):
                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                os.mkdir(cat_dir)
            img_pil = image.load_img(img_filename)
            img = image.img_to_array(img_pil)
    
                
            

            imgPath = os.path.join(currOut, img_filename)
           # train with context around box
            
            contextMultWidth = 0.15
            contextMultHeight = 0.15
            
            wRatio = float(box[2]) / img.shape[0]
            hRatio = float(box[3]) / img.shape[1]
            
            if wRatio < 0.5 and wRatio >= 0.4:
                contextMultWidth = 0.2
            if wRatio < 0.4 and wRatio >= 0.3:
                contextMultWidth = 0.3
            if wRatio < 0.3 and wRatio >= 0.2:
                contextMultWidth = 0.5
            if wRatio < 0.2 and wRatio >= 0.1:
                contextMultWidth = 1
            if wRatio < 0.1:
                contextMultWidth = 2
                
            if hRatio < 0.5 and hRatio >= 0.4:
                contextMultHeight = 0.2
            if hRatio < 0.4 and hRatio >= 0.3:
                contextMultHeight = 0.3
            if hRatio < 0.3 and hRatio >= 0.2:
                contextMultHeight = 0.5
            if hRatio < 0.2 and hRatio >= 0.1:
                contextMultHeight = 1
            if hRatio < 0.1:
                contextMultHeight = 2
            
            
            widthBuffer = int((box[2] * contextMultWidth) / 2.0)
            heightBuffer = int((box[3] * contextMultHeight) / 2.0)

            r1 = box[1] - heightBuffer
            r2 = box[1] + box[3] + heightBuffer
            c1 = box[0] - widthBuffer
            c2 = box[0] + box[2] + widthBuffer

            if r1 < 0:
                r1 = 0
            if r2 > img.shape[0]:
                r2 = img.shape[0]
            if c1 < 0:
                c1 = 0
            if c2 > img.shape[1]:
                c2 = img.shape[1]

            if r1 >= r2 or c1 >= c2:
                continue

            subImg = img[r1:r2, c1:c2, :]
            subImg = cv2.resize(subImg, params.target_img_size)
            allResults.append(subImg)
            cv2.imwrite(currOut, subImg)
            cat_value = params.category_names.index(category) 
            allCats.append(to_categorical(cat_value, params.num_labels))
            return np.asarray(allResults), allCats
        else:
            return None, category

def load_from_full(data_dir,train_dir='train/'):
    if not os.path.exists(os.path.join(train_dir,'fmow_all_filenames.npy')):
        all_jsons = []
        counter = 0
        cats = os.listdir(data_dir)
        for cat in cats:
            cat_folder = os.path.join(data_dir,cat)
            folders = os.listdir(cat_folder)
            for folder in folders:
                direc = os.path.join(cat_folder,folder)
                for filename in os.listdir(direc):
                    if filename.endswith('.json'):
                        final_path = os.path.join(direc,filename)
                        all_jsons.append(final_path)
                        x,y = prep_image(final_path,train_dir)
        all_jsons = np.asarray(all_jsons)
        random.shuffle(all_jsons)
        np.save((os.path.join(train_dir,'fmow_all_filenames.npy')), all_jsons)
    else:
        all_jsons = np.load(os.path.join(train_dir,'fmow_all_filenames.npy'))
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    data_dir = params.directories['dataset']
    load_from_full(data_dir)



