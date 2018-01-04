#!/bin/env python

""" Code for developing and attacking a simple street sign detector.

REFERENCES:
 - LISA data set http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
"""


import os
import random
import math
import json
from PIL import Image
import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import keras

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, CarliniWagnerL2
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval, batch_eval

import subimage


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


#-------------------------------------------------------------------------------
# Helper/utility functions
#-------------------------------------------------------------------------------

def to_categorical(y, num_classes, dtype=np.float32, smooth=False):
    """ Converts a vector of integer class labels into a one-hot matrix representation.

      e.g.  [1 3 7] becomes:

          [0 1 0 0 0 0 0 0 0 0 0 ;
           0 0 0 1 0 0 0 0 0 0 0 ;
           0 0 0 0 0 0 0 1 0 0 0 ]
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



def categorical_matrix(y_scalar, num_copies, num_classes, *args, **kargs):
    """ Creates a matrix of one-hot target class labels
        (for use with targeted attacks).  

        e.g. categorical_matrix(1,3,10) is

          [ 0 1 0 0 0 0 0 0 0 0 ;
            0 1 0 0 0 0 0 0 0 0 ;
            0 1 0 0 0 0 0 0 0 0 ]
    """
    y_tgt = y_scalar * np.ones((num_copies,), dtype=np.int32)
    return to_categorical(y_tgt, num_classes, *args, **kargs)



def calc_acc(y_true_OH, y_hat_OH):
    """ Computes classification accuracy from a pair of one-hot estimates/truth.

    It doesn't really matter if the arguments are one-hot; just need the
    argmax along dimension 1 gives the estimated class label.

    """
    is_correct = np.argmax(y_hat_OH, axis=1) == np.argmax(y_true_OH, axis=1)
    return 1. * np.sum(is_correct) / y_hat_OH.shape[0]


def makedirs_if_needed(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



def run_in_batches(sess, x_tf, y_tf, output_tf, x_in, y_in, batch_size):
    """ 
     Runs data through a CNN one batch at a time; gathers all results
     together into a single tensor.  This assumes the output of each
     batch is tensor-like.

        sess      : the tensorflow session to use
        x_tf      : placeholder for input x
        y_tf      : placeholder for input y
        output_tf : placeholder for CNN output
        x_in      : data set to process (numpy tensor)
        y_in      : associated labels (numpy, one-hot encoding)
        batch_size : minibatch size (scalar)

    """
    n_examples = x_in.shape[0]  # total num. of objects to feed

    # determine how many mini-batches are required
    nb_batches = int(math.ceil(float(n_examples) / batch_size))
    assert nb_batches * batch_size >= n_examples

    out = []
    with sess.as_default():
        for start in np.arange(0, n_examples, batch_size):
            # the min() stuff here is to handle the last batch, which may be partial
            end = min(n_examples, start + batch_size)
            start_actual = min(start, n_examples - batch_size)

            feed_dict = {x_tf : x_in[start_actual:end], y_tf : y_in[start_actual:end]}
            output_i = sess.run(output_tf, feed_dict=feed_dict)

            # the slice is to avoid any extra stuff in last mini-batch,
            # which might not be entirely "full"
            skip = start - start_actual
            output_i = output_i[skip:]
            out.append(output_i)

    out = np.concatenate(out, axis=0)
    assert(out.shape[0] == n_examples)
    return out


#-------------------------------------------------------------------------------
# LISA-CNN codes
#-------------------------------------------------------------------------------

def load_lisa_data(with_context=True):
    """
    Loads LISA data set.
    
    FLAGS:
        annotation_file: main LISA annotation file

    RETURNS:
        xtrain: numpy array of training data
        ytrain: numpy array of traning labels
        xtest: numpy array of test data
        ytest: numpy array of test labels
    """
    annotation_file = '~/Data/LISA/allAnnotations.csv'
    cache_fn = './output/lisa_data.npz'

    # Create the dataset if it does not already exist.
    # Note that we create it once, up front, so that we
    # can ensure a consistent train/test split across all experiments.
    if not os.path.exists(cache_fn):
        print('[load_lisa_data]: Extracting sign images from video frames...please wait...')

        si = subimage.parse_LISA(annotation_file)
        train_idx, test_idx = si.train_test_split(.17, max_per_class=500)
        print(si.describe(train_idx))
        print(si.describe(test_idx))

        #
        # signs *without* extra context
        #
        x_train, y_train = si.get_subimages(train_idx, (32,32), 0.0)
        x_train = np.array(x_train) # list -> tensor
        y_train = np.array(y_train)

        x_test, y_test = si.get_subimages(test_idx, (32,32), 0.0)
        x_test = np.array(x_test) # list -> tensor
        y_test = np.array(y_test)

        #
        # signs *with* extra context
        #
        sz = 38
        pct = (38-32)/38.
        x_train_c, y_train_c = si.get_subimages(train_idx, (sz,sz), pct)
        x_train_c = np.array(x_train_c)
        y_train_c = np.array(y_train_c)

        x_test_c, y_test_c = si.get_subimages(test_idx, (sz,sz), pct)
        x_test_c = np.array(x_test_c)
        y_test_c = np.array(y_test_c) 

        # (optional) rescale
        # with the default hyperparameters, this actually hurts performance...
        if 0:
            x_train /= 255.
            x_test /= 255.
            x_train_c /= 255.
            x_test_c /= 255.

        # save for quicker reload next time
        makedirs_if_needed(os.path.dirname(cache_fn))
        np.savez(cache_fn, train_idx=train_idx,  
                           test_idx=test_idx, 
                           x_train=x_train, 
                           y_train=y_train, 
                           x_test=x_test, 
                           y_test=y_test,
                           x_train_context=x_train_c, 
                           y_train_context=y_train_c,
                           x_test_context=x_test_c, 
                           y_test_context=y_test_c)

    f = np.load(cache_fn)

    if with_context:
        x_train = f['x_train_context']
        y_train = f['y_train_context']
        x_test = f['x_test_context']
        y_test = f['y_test_context']
    else:
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']


    for string, data in zip(['train data', 'train labels', 'test data', 'test labels'], [x_train, y_train, x_test, y_test]):
        print('[load_lisa_data]: ', string, data.shape, data.dtype, np.min(data), np.max(data))

    return x_train, y_train, x_test, y_test



def data_lisa(with_context):
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
    # NOTE: it would appear the LISA annotation extraction code introduces some 
    #       label noise.  Therefore, we do the extraction ourselves.
    X_train, Y_train, X_test, Y_test = load_lisa_data(with_context)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Note: we moved the affine rescaling directly into the network

    # Downsample to promote class balance during training
    #X_train, Y_train = downsample_data_set(X_train, Y_train, per_class_limit)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape: ', X_test.shape)

    all_labels = np.unique(Y_train)
    print('There are %d unique classes:' % all_labels.size)
    for yi in all_labels:
        print('  y=%d : %d' % (yi, np.sum(Y_train==yi)))

    Y_train = to_categorical(Y_train, FLAGS.nb_classes)
    Y_test = to_categorical(Y_test, FLAGS.nb_classes)

    return X_train, Y_train, X_test, Y_test



def make_lisa_cnn(sess, batch_size, dim):
    """  Creates a simple sign classification network.
    
    Note that the network produced by cnn_model() is fairly weak.
    For example, on CIFAR-10 it gets 60-something percent accuracy,
    which substantially below state-of-the-art.

    Note: it is not required here that dim be the same as the
    CNN input spatial dimensions.  In these cases, the
    caller is responsible for resizing x to make it
    compatible with the model (e.g. via random crops).
    """
    num_classes=48  # usually only use 17 vs 48 classes, but it doesn't hurt to have room for 48
    num_channels=1
    x = tf.placeholder(tf.float32, shape=(batch_size, dim, dim, num_channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    # XXX: set layer naming convention explicitly? 
    #      Otherwise, names depend upon when model was created...
    model = cnn_model(img_rows=32, img_cols=32, channels=num_channels, nb_classes=num_classes)

    return model, x, y




def train_lisa_cnn(sess, cnn_weight_file):
    """ Trains the LISA-CNN network.
    """
    X_train, Y_train, X_test, Y_test = data_lisa(with_context=True)

    model, x, y = make_lisa_cnn(sess, FLAGS.batch_size, X_train.shape[1])

    # construct an explicit predictions variable
    x_crop = tf.random_crop(x, (FLAGS.batch_size, 32, 32, X_train.shape[-1]))
    model_output = model(x_crop)

    def evaluate():
        # Evaluate accuracy of the lisaCNN model on clean test examples.
        preds = run_in_batches(sess, x, y, model_output, X_test, Y_test, FLAGS.batch_size)
        print('test accuracy: ', calc_acc(Y_test, preds))

    # Note: model_train() will add some new (Adam-related) variables to the graph
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train(sess, x, y, model_output, X_train, Y_train, evaluate=evaluate, args=train_params)

    saver = tf.train.Saver()
    save_path = saver.save(sess, cnn_weight_file)
    print("Model was saved to " +  cnn_weight_file)



def attack_lisa_cnn(sess, cnn_weight_file, y_target=None):
    """ Generates AE for the LISA-CNN.
        Assumes you have already run train_lisa_cnn() to train the network.
    """
    #--------------------------------------------------
    # data set prep
    #--------------------------------------------------
    # Note: we load the version of the data *without* extra context
    X_train, Y_train, X_test, Y_test = data_lisa(with_context=False)

    # Create one-hot target labels (needed for targeted attacks only)
    if y_target is not None:
        Y_target_OB = categorical_matrix(y_target, FLAGS.batch_size, Y_test.shape[1])
        Y_target = categorical_matrix(y_target, Y_test.shape[0], Y_test.shape[1])
    else:
        Y_target_OB = None
        Y_target = None

    #--------------------------------------------------
    # Initialize model that we will attack
    #--------------------------------------------------
    model, x_tf, y_tf = make_lisa_cnn(sess, FLAGS.batch_size, X_train.shape[1])
    model_output = model(x_tf)
    # TODO: wrap in cleverhans Model object

    saver = tf.train.Saver()
    saver.restore(sess, cnn_weight_file)

    #--------------------------------------------------
    # Before attacking, verify performance is good on clean data
    #--------------------------------------------------
    predictions = run_in_batches(sess, x_tf, y_tf, model(x_tf), X_test, Y_test, FLAGS.batch_size)
    acc_clean = 100. * calc_acc(Y_test, predictions)
    print('[info]: accuracy on clean test data: %0.2f' % acc_clean)
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(predictions, axis=1)))

    #--------------------------------------------------
    # Fast Gradient Attack
    #--------------------------------------------------
    # symbolic representation of attack
    attack = FastGradientMethod(model, sess=sess)
    x_adv_tf = attack.generate(x_tf, eps=FLAGS.epsilon, y_target=Y_target_OB, clip_min=0, clip_max=255)

    #
    # Run the attack (targeted or untargeted)
    # on the test data.
    #
    if Y_target is not None:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
    else:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

    #
    # Evaluate the AE. 
    # Currently using the same model we originally attacked.
    #
    model_eval = model
    preds_tf = model_eval(x_tf)
    preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
    print('Test accuracy on adversarial examples: %0.3f' % calc_acc(Y_test, preds))
    print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

    #--------------------------------------------------
    # Iterative attack
    #--------------------------------------------------
    attack = BasicIterativeMethod(model, sess=sess)
    x_adv_tf = attack.generate(x_tf, eps=FLAGS.epsilon, 
                                     eps_iter=FLAGS.epsilon/2, 
                                     nb_iter=100,
                                     y_target=Y_target_OB, 
                                     clip_min=0, 
                                     clip_max=255)

    #
    # Run the attack (targeted or untargeted)
    # on the test data.
    #
    if Y_target is not None:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
    else:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

    #
    # Evaluate the AE. 
    # Currently using the same model we originally attacked.
    #
    model_eval = model
    preds_tf = model_eval(x_tf)
    preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
    print('Test accuracy on adversarial examples: %0.3f' % calc_acc(Y_test, preds))
    print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

    return # TEMP

    #--------------------------------------------------
    # C&W ell-2
    #--------------------------------------------------
    attack = CarliniWagnerL2(model, sess=sess)
    x_adv_tf = attack.generate(x_tf, confidence=.1, y_target=Y_target_OB)

    if Y_target is not None:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
    else:
        X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

    model_eval = model
    preds_tf = model_eval(x_tf)
    preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
    print('Test accuracy on adversarial examples: %0.3f' % calc_acc(Y_test, preds))
    print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))


    pdb.set_trace() # TEMP
    if FLAGS.save_adv_img:

        # MJP: This code breaks when running targeted attacks.
        #      May be due to experimental nature of this API...
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
            orig_im.save(os.path.join(out_dir, 'orig', 'image'+str(total_images)+'.jpg'))
            adv_pil.save(os.path.join(out_dir, 'adv', 'image'+str(total_images)+'.jpg'))
                
        print(float(correct_predictions)/total_images)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def main(argv=None):
    # Set TF random seed to improve reproducibility
    np.random.seed(1066)
    tf.set_random_seed(1246)

    # the CNN weight file
    cnn_weight_file = os.path.join(FLAGS.train_dir, FLAGS.filename)

    class_map = subimage.LISA_17_CLASS_MAP

    with tf.Session() as sess:
      if not os.path.exists(FLAGS.train_dir) or not tf.train.checkpoint_exists(cnn_weight_file) or FLAGS.force_retrain:
          print("Training LISA-CNN")
          train_lisa_cnn(sess, cnn_weight_file)
      else:
          print("Attacking LISA-CNN")
          attack_lisa_cnn(sess, cnn_weight_file, y_target=class_map['speedLimit45'])
          #attack_lisa_cnn(sess, cnn_weight_file, y_target=None)


if __name__ == '__main__':
    app.run()
