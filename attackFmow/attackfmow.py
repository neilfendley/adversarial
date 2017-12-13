import pdb
import sys
sys.path.append('DenseNet')
from cleverhans.attacks import FastGradientMethod
import pandas as pd
from sklearn.metrics import confusion_matrix
import random
import cv2
import json
from keras import backend as K
import params
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.layers import Dense,Input,merge,Flatten,Dropout,LSTM
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import densenet
import numpy as np
import os
from keras.preprocessing import image
from cleverhans.utils_keras import KerasModelWrapper

def compare(model, xTest, yTest):
    predictions = np.argmax(model.predict(xTest, batch_size=32), axis = 1)
    truths = np.argmax(yTest, axis = 1)
    total = 0
    corr = 0
    for truth, pred in zip(truths, predictions):
        if truth == pred:
            corr += 1
        total +=1
    return float(corr)/ total, total

def cnn_model():
    input_tensor = Input(shape=(params.target_img_size[0],params.target_img_size[1],params.num_channels))
    baseModel = densenet.DenseNetImageNet161(input_shape=(params.target_img_size[0], params.target_img_size[1], params.num_channels), include_top=False, input_tensor=input_tensor)
    modelStruct = baseModel.layers[-1].output

    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc2')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    predictions = Dense(params.num_labels, activation='softmax')(modelStruct)
    model = Model(inputs=[baseModel.input], outputs=predictions)

    for i,layer in enumerate(model.layers[-5:]):
        layer.trainable = True
    for i,layer in enumerate(model.layers[:-5]):
        layer.trainable = False

    return model

def predict_data():
    model = cnn_model()
    model.load_weights('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
    folders = os.listdir(data_dir)
    all_paths = prep_filelist()
    preds_all = np.zeros((0,63))
    y_all = np.zeros((0,63))
    counter = 0
    hits = 0
    for img_path in all_paths:
        img = cv2.imread(img_path).astype(np.float32)
        category = img_path.split('/')[-2]
        xtest = np.expand_dims(img, axis=0)
        xtest = imagenet_utils.preprocess_input(xtest)
        xtest = xtest / 255.0
        y_test = to_categorical(params.category_names.index(category), params.num_labels)

        pred = model.predict(xtest, batch_size=1)
        preds_all = np.concatenate((preds_all, pred), axis=0) 
        y_all = np.concatenate((y_all, y_test), axis=0)
        if np.argmax(y_test) == np.argmax(pred):
            hits +=1
        counter += 1
        if counter % 1000 == 0:
            assert y_all.shape == preds_all.shape
            print (float(hits) / counter)
            np.save(os.path.join(data_dir, 'y_all_temp.npy'), y_all)
            np.save(os.path.join(data_dir, 'preds_all_temp.npy'), preds_all)
    np.save(os.path.join(data_dir, 'y_all.npy'), y_all)
    np.save(os.path.join(data_dir, 'preds_all.npy'), preds_all)

def conf_matrix():
    y_all = np.load(os.path.join(params.directories['data'], 'y_all.npy'))
    preds_all = np.load(os.path.join(params.directories['data'],'preds_all.npy'))
    preds = np.argmax(preds_all, axis = 1)
    y_test = np.argmax(y_all, axis=1)
    conf = confusion_matrix(y_test, preds)
    np.save(os.path.join(params.directories['data'],'confusion_matrix.npy'), conf)
    df = pd.DataFrame(conf)
    df.to_pickle(os.path.join(params.directories['data'],'confusion_matrix'))

def prep_filelist():
    all_paths = []
    if os.path.exists(os.path.join(params.directories['cache'], 'filepaths.npy')):
        all_paths = np.load(os.path.join(params.directories['cache'], 'filepaths.npy'))
    else:
        folders = os.listdir(params.directories['data'])
        for folder in folders:
            folder_path = os.path.join(params.directories['data'], folder)
            if os.path.isdir(folder_path):
                img_files = os.listdir(folder_path)
                for img_file in img_files:
                    if img_file.endswith('.jpg'):
                        all_paths.append(os.path.join(folder_path, img_file))
        np.save(os.path.join(params.directories['cache'], 'filepaths.npy'), all_paths)
    print("[ INFO ]:  Prepped " + str(len(all_paths)) + " image paths")
    return all_paths

def prep_adv_testing(model, filepaths, num_adv=1000):
    i = 0
    xTest = np.zeros((0,224,224,3))
    yTest = np.zeros((0,63))
    print("[ INFO ]:  We are loading correctly detected images")

    if os.path.exists(os.path.join(params.directories['cache'],'xadvtest.npy')):
        xTest = np.load(os.path.join(params.directories['cache'],'xadvtest.npy'))
        yTest = np.load(os.path.join(params.directories['cache'],'yadvtest.npy'))
    while xTest.shape[0] < num_adv:
        img_path = filepaths[i]
        i += 1
        img = cv2.imread(img_path).astype(np.float32)
        category = img_path.split('/')[-2]
        xtest = np.expand_dims(img, axis=0)
        xtest = imagenet_utils.preprocess_input(xtest)
        xtest = xtest / 255.0
        y_test = to_categorical(params.category_names.index(category), params.num_labels)

        pred = model.predict(xtest, batch_size=1)

        if np.argmax(y_test) == np.argmax(pred):
            xTest = np.concatenate((xTest, xtest), axis = 0)
            yTest = np.concatenate((yTest, y_test), axis = 0)
        if i % 1000 == 0:
            print ("[ STATUS]:  On image " + str(i) + ', ' + str(xTest.shape[0]) + " good images loaded")

    np.save(os.path.join(params.directories['cache'],'xadvtest.npy'), xTest)
    np.save(os.path.join(params.directories['cache'],'yadvtest.npy'), yTest)
    print ("[ Info ]: Finished loading good clean examples")
    return xTest, yTest
    

def adv_fgsm():
    sess = tf.Session()
    K.set_learning_phase(0)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer()) 
    data_dir = 'train/'
    model = cnn_model()
    model.load_weights('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
   
    all_paths = prep_filelist()
    xTest, yTest = prep_adv_testing(model,all_paths)

    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 63))
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    eps = [0.0,0.01, 0.03, 0.04, 0.1]
    for ep in eps:
        fgsm_params = {'eps': ep}
        adv_x = fgsm.generate(x, **fgsm_params)
        # Consider the attack to be constant
        #adv_x = tf.stop_gradient(adv_x)
        eval_par = {'batch_size': 32}
        counter = 0
        hits = 0
        for i in range(len(xTest)):
            img_clean = np.expand_dims(xTest[i], axis=0)
            if ep == 0:
                img = img_clean
            else:
                img = adv_x.eval(session=sess, feed_dict={x:img_clean})
            cat = yTest[i]
            cat_input = np.expand_dims(cat, axis=0)
            pred = model.predict(img, batch_size=1)
            if np.argmax(pred) == np.argmax(cat_input):
                hits += 1
            counter += 1
        print("[ Info ]: The accuracy on eps " + str(ep) + ': ' +str(float(hits)/counter))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    adv_fgsm()
    

 
if __name__ == "__main__":
    main()
