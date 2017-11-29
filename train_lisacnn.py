#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
import keras
from keras import backend
from keras.utils import np_utils
import pdb
from sklearn.metrics import confusion_matrix
import pandas as pd
import cv2
import json
import os
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import numpy as np
from cleverhans.attacks import fgsm
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', './tmp_pub', 'Directory storing the saved model.')
flags.DEFINE_string('data_dir','/home/fendlnm1/Fendley/street_signs/signDatabase/annotations', 'The Directory in which the extra lisadataset is')
flags.DEFINE_string(
    'filename', '/lisacnn.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 60, 'Number of epochs to train model')
flags.DEFINE_integer('nb_classes', 48, 'Number of classes')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
flags.DEFINE_bool('DO_CONF', False, 'Generate the confusion matrix on the test set')
flags.DEFINE_bool('DO_ADV', False, 'Generate the adversarial examples on the test set')
flags.DEFINE_bool('force_retrain', True, 'Ignore if you have already trained a model')

def load_lisa_data():
    """
    Function to load the lisa datset from specified directory, or will re-load precomputed numpy arrays
    FLAGS:
        train_dir: location to save data split np arrays and model checkpoint
        data_dir: locataion of the extracted lisa data on local machine
    RETURNS:
        xtrain: numpy array of training data
        ytrain: numpy array of traning labels
        xtest: numpy array of test data
        ytest: numpy array of test labels
    """
       # Check if you have created the numpy arrays already
    if os.path.exists(os.path.join(FLAGS.train_dir,'ytest.npy')):
        X_train = np.load(os.path.join(FLAGS.train_dir, 'xtrain.npy'))
        Y_train = np.load(os.path.join(FLAGS.train_dir, 'ytrain.npy'))
        X_test = np.load(os.path.join(FLAGS.train_dir, 'xtest.npy'))
        Y_test = np.load(os.path.join(FLAGS.train_dir, 'ytest.npy'))
    else:
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        counter = 0
        class_counter = 0
        classes = 48
        pic_list = os.listdir(FLAGS.data_dir)
        # This is so we have a constant class mapping
        mapping = 'class_mapping.json'
        if os.path.exists(mapping):
            with open(mapping, 'r') as fr:
                cat_dict = json.load(fr)
        else:
            print("[ ERROR ]: No class mapping file")
            cat_dict = {}
        cat_total = {}
        cat_test = {}
        total_img = len(pic_list)
        #Randomize the train test split
        random.shuffle(pic_list)
        # Read in the image, resize it  to (32, 32) and parse the class
        for pic_url in pic_list:
            x = cv2.imread(os.path.join(FLAGS.data_dir,pic_url))
            x_resize = cv2.resize(x.copy(), (32,32))
            cat = pic_url.split('_')[1]
            if cat not in cat_total.keys():
                cat_total[cat] = 0
                cat_test[cat] = 0
            y = cat_dict[cat]
            if cat_total[cat] == 0 or (float(cat_test[cat]) / cat_total[cat] > .1):
                X_train.append(x_resize)
                Y_train.append(y)
            else:
                X_test.append(x_resize)
                Y_test.append(y)
                cat_test[cat] += 1
            cat_total[cat] += 1

            counter += 1
        os.mkdir(FLAGS.train_dir)
        np.save(os.path.join(FLAGS.train_dir, 'xtrain.npy'), X_train)
        np.save(os.path.join(FLAGS.train_dir, 'ytrain.npy'), Y_train)
        np.save(os.path.join(FLAGS.train_dir, 'xtest.npy'), X_test)
        np.save(os.path.join(FLAGS.train_dir, 'ytest.npy'), Y_test)
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)

def data_lisa():
    """
    Funtion to read in the data prepared by the lisa dataset
    The train test split will be randomly generated, or loaded if you have a /tmp 
    split saved already
    Returns:
        xtrain: numpy array of the training data 
        ytrain: categorical array of training labels
        xtest: numpy array of the test data
        ytest: numpy array of the test labels
    
    """

    X_train, Y_train, X_test, Y_test = load_lisa_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    Y_train = np_utils.to_categorical(Y_train, FLAGS.nb_classes)
    Y_test = np_utils.to_categorical(Y_test, FLAGS.nb_classes)
    

    return X_train, Y_train, X_test, Y_test

def main(argv=None):

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1246)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    
    X_train, Y_train, X_test, Y_test = data_lisa()

    # Label Smoothing 

    #label_smooth = .1
    #Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    model = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=FLAGS.nb_classes)
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()
    def evaluate():
        # Evaluate the accuracy of the lisaCNN model on legitimate test
        # examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Training 
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    # If we have already trained a model, load it 

    save_string = os.path.join(FLAGS.train_dir + FLAGS.filename)
    if tf.train.checkpoint_exists(save_string) and not FLAGS.force_retrain:
        saver.restore(sess, save_string)
    else:
        model_train(sess, x, y, predictions, X_train, Y_train,evaluate=evaluate, args=train_params)
    save_path = saver.save(sess,save_string)
    print("Model was saved to " +  save_string)
    if FLAGS.DO_CONF:
        preds = np.argmax(predictions.eval(session=sess,feed_dict={x : X_test}),axis=1)
        y_test = np.argmax(Y_test, axis=1)
        conf = confusion_matrix(y_test, preds)
        wrong = 0
        total = 0
        for y,yt in zip(y_test, preds):
            if y != yt:
                wrong += 1
            total += 1
        acc = 1 - (float(wrong) / total)
        df = pd.DataFrame(conf)

    if FLAGS.DO_ADV:
        # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
        adv_x = fgsm(x, predictions, eps=0.3)
        eval_params = {'batch_size': FLAGS.batch_size}
        X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)

        # Evaluate the accuracy of the CIFAR10 model on adversarial examples
        accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test,
                              args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy))

        print("Repeating the process, using adversarial training")
        # Redefine TF model graph
        model_2 = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=FLAGS.nb_classes)
        predictions_2 = model_2(x)
        adv_x_2 = fgsm(x, predictions_2, eps=0.3)
        predictions_2_adv = model_2(adv_x_2)

        def evaluate_2():
            # legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

            # adversarial examples
            accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test,
                                      Y_test, args=eval_params)
            print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

        # Perform adversarial training
        model_train(sess, x, y, predictions_2, X_train, Y_train,
                    predictions_adv=predictions_2_adv, evaluate=evaluate_2,
                    args=train_params)

        # Evaluate the accuracy of the CIFAR10 model on adversarial examples
        accuracy = model_eval(sess, x, y, predictions_2_adv, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on adversarial examples: ' + str(accuracy))


if __name__ == '__main__':
    app.run()
