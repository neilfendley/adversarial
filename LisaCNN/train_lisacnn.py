#!/usr/bin/env python


import random
import keras
#from keras.utils import np_utils
import pdb
from sklearn.metrics import confusion_matrix
import pandas as pd
from PIL import Image
import json
import os
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import numpy as np
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'output', 'Directory storing the saved model and other outputs (e.g. adversarial images).')
flags.DEFINE_string('data_dir','/home/neilf/Fendley/data/signDatabase/annotations', 'The Directory in which the extra lisadataset is')
flags.DEFINE_string(
    'filename', 'lisacnn.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 60, 'Number of epochs to train model')
flags.DEFINE_integer('nb_classes', 48, 'Number of classes')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
flags.DEFINE_float('epsilon', 20, 'FGSM perturbation constraint')
flags.DEFINE_bool('DO_CONF', False, 'Generate the confusion matrix on the test set')
flags.DEFINE_bool('DO_ADV', True, 'Generate the adversarial examples on the test set')
flags.DEFINE_bool('force_retrain', False, 'Ignore if you have already trained a model')
flags.DEFINE_bool('save_adv_img', True, 'Save the adversarial images generated on the test set')



def to_categorical(y, num_classes, dtype=np.float32, smooth=False):
    """ Converts a vector of integer class labels into a one-hot matrix representation.
    """
    out = np.zeros((y.size, num_classes), dtype=dtype)
    for ii in range(y.size):
        out[ii,y[ii]] = 1

    if smooth:
      # put .95 weight on true class, divide rest among other classes
      nz_value = 0.05 / (num_classes-1)
      out[out==0] = nz_value
      out[out==1] = 0.95

    return out


def makedirs_if_needed(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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
            x_pil = Image.open(os.path.join(FLAGS.data_dir,pic_url))
            x = x_pil.resize((32,32), Image.ANTIALIAS)
            x_resize = np.array(x)
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

    # Note: we now put the affine rescaling directly into the network

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print('Y_train shape:', Y_train.shape)
    #Y_train = np_utils.to_categorical(Y_train, FLAGS.nb_classes)
    #Y_test = np_utils.to_categorical(Y_test, FLAGS.nb_classes)
    Y_train = to_categorical(Y_train, FLAGS.nb_classes)
    Y_test = to_categorical(Y_test, FLAGS.nb_classes)
    
    return X_train, Y_train, X_test, Y_test



def make_lisa_cnn(sess):
    """ Note that the network produced by cnn_model() is fairly weak.
        For example, on CIFAR-10 it gets 60-something percent accuracy,
        which substantially below state-of-the-art.
    """
    num_classes=48
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))

    # This affine transformation may not be strictly necessary
    x = (x / 255.) - 0.5

    # TODO: fix naming convention explicitly? 
    # otherwise, names depend upon when model was created...
    model = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=num_classes)

    return model, x, y



def train_lisa_cnn(sess, save_string):
    """ Trains the LISA-CNN network.
    """
    X_train, Y_train, X_test, Y_test = data_lisa()   # load data
    model, x, y = make_lisa_cnn(sess)

    def evaluate():
        # Evaluate the accuracy of the lisaCNN model on legitimate test
        # examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, model(x), X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }

    # Note: i believe this will add some new (Adam-related) variables to the graph
    model_train(sess, x, y, model(x), X_train, Y_train, evaluate=evaluate, args=train_params)

    saver = tf.train.Saver()
    save_path = saver.save(sess, save_string)
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



def attack_lisa_cnn(sess, save_string):
      # Adversarial attack
      X_train, Y_train, X_test, Y_test = data_lisa()

      # restore model
      # TODO: do we need to specify test mode?
      model, x, y = make_lisa_cnn(sess)
      saver = tf.train.Saver()
      saver.restore(sess, save_string)

      # create attack
      fgsm = FastGradientMethod(model, sess=sess)
      x_adv = fgsm.generate(x, eps=FLAGS.epsilon)

      # create a model to attack.
      # In this case, it is the same model used to generate AE (LISA-CNN)
      model_tgt = model
      pred_tgt = model_tgt(x_adv)

      accuracy = model_eval(sess, x, y, pred_tgt, X_test, Y_test, args={'batch_size' : FLAGS.batch_size})
      print('Test accuracy on adversarial examples: ' + str(accuracy))

      if FLAGS.save_adv_img:
          adv_part = sess.partial_run_setup([x_adv, pred_tgt], [x])
          adv_out = sess.partial_run(adv_part, x_adv, feed_dict={x:X_test})
          preds_adv = sess.partial_run(adv_part, pred_tgt)

          #  Define the directories to save adversarial images
          #
          # MJP: updating directory layout slightly.
          #
          fooled_adv_dir = os.path.join(FLAGS.train_dir, 'images/fooled_FGSM%0.2f/' % FLAGS.epsilon)
          correct_predicted_dir = os.path.join(FLAGS.train_dir, 'images/not_fooled_FGSM%0.2f/' % FLAGS.epsilon)
          for dirname in [fooled_adv_dir, correct_predicted_dir]:
              makedirs_if_needed(os.path.join(dirname, 'adv'))
              makedirs_if_needed(os.path.join(dirname, 'orig'))

          #  Keep track of the total images, and how many are correctly detected
          total_images = 0
          correct_predictions = 0
          for i in range(len(X_test)):
              adv_img = adv_out[i] + .5
              adv_img *= 255
              adv_pred = preds_adv[i]
              adv_pil = Image.fromarray(adv_img.astype('uint8'))
              truth = Y_test[i]
              if np.argmax(adv_pred) == np.argmax(truth):
                  out_dir = correct_predicted_dir
                  #adv_pil.save(os.path.join(correct_predicted_dir,'adversarial_image'+str(total_images)+'.jpg'))
                  correct_predictions += 1
              else:
                  out_dir = fooled_adv_dir
                  #adv_pil.save(os.path.join(fooled_adv_dir,'adversarial_image'+str(total_images)+'.jpg'))
              total_images += 1

              orig_im = Image.fromarray(X_test[i].astype('uint8'))
              #orig_im.save(os.path.join(orig_dir,'original_image'+str(total_images)+'.jpg'))
              orig_im.save(os.path.join(out_dir, 'orig', 'image'+str(total_images)+'.jpg'))
              adv_pil.save(os.path.join(out_dir, 'adv', 'image'+str(total_images)+'.jpg'))
                
          print(float(correct_predictions)/total_images)


def main(argv=None):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1246)

    save_string = os.path.join(FLAGS.train_dir, FLAGS.filename)

    with tf.Session() as sess:
      if not os.path.exists(FLAGS.train_dir) or not tf.train.checkpoint_exists(save_string) or FLAGS.force_retrain:
          print("Training LISA-CNN")
          train_lisa_cnn(sess, save_string)
      else:
          print("Attacking LISA-CNN")
          attack_lisa_cnn(sess, save_string)


if __name__ == '__main__':
    app.run()
