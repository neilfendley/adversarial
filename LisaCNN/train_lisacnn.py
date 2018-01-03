#!/bin/env python


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

from cleverhans.attacks import FastGradientMethod
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


def calc_acc(y_true_OH, y_hat_OH):
    is_correct = np.argmax(y_hat_OH, axis=1) == np.argmax(y_true_OH, axis=1)
    return 1. * np.sum(is_correct) / y_hat_OH.shape[0]


#def downsample_data_set(x, y, max_per_class):
#    """ Downsamples a data set so that the maximum number of 
#        representatives from any one class is max_per_class.
#
#        x : a tensor of feature data with shape (n, ...) where n is the number of examples.
#        y : a vector of integer class labels with shape (n,)
#    """
#    y_all = np.unique(y)
#    to_keep = np.zeros((y.size,), dtype=np.bool)
#
#    for yi in y_all:
#        indices = np.nonzero(y == yi)[0]
#        if len(indices) > max_per_class:
#            indices = np.random.choice(indices, max_per_class, replace=False)
#        to_keep[indices] = True
#
#    return x[to_keep,...], y[to_keep]



def makedirs_if_needed(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



def load_lisa_data(with_context=True, grayscale=True):
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
    cache_fn = 'lisa_data.npz'

    # Create the dataset if it does not already exist.
    # Note that we create it once, up front, so that we
    # can ensure a consistent train/test split across all experiments.
    if not os.path.exists(cache_fn):
        print('[LLD]: Extracting sign images from video frames...please wait...')

        si = subimage.parse_LISA(annotation_file)
        train_idx, test_idx = si.train_test_split(.17, max_per_class=500)
        print(si.describe(train_idx))
        print(si.describe(test_idx))

        # extract data *without* context
        x_train, y_train = si.get_subimages(train_idx, (32,32), 0.0)
        x_train = np.array(x_train) # list -> tensor
        y_train = np.array(y_train)

        x_test, y_test = si.get_subimages(test_idx, (32,32), 0.0)
        x_test = np.array(x_test) # list -> tensor
        y_test = np.array(y_test)

        # extract data *with* context
        sz = 38
        pct = (38-32)/38.
        x_train_c, y_train_c = si.get_subimages(train_idx, (sz,sz), pct)
        x_train_c = np.array(x_train_c)
        y_train_c = np.array(y_train_c)

        x_test_c, y_test_c = si.get_subimages(test_idx, (sz,sz), pct)
        x_test_c = np.array(x_test_c)
        y_test_c = np.array(y_test_c) 

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

    #
    # UPDATE:
    # Make test size a multiple of batch size due to CH issues in model_eval
    # TODO: can we remove this  now that we don't require model_eval?
    #
    rem = np.mod(x_test.shape[0], FLAGS.batch_size)
    if rem != 0:
        print('WARNING: trucating eval size by %d' % rem)
        x_test = x_test[:-rem,...]
        y_test = y_test[:-rem,...]
        print(x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test



def old_load_lisa_data():
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
    num_classes=48  # 17 vs 48
    num_channels=1
    x = tf.placeholder(tf.float32, shape=(batch_size, dim, dim, num_channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    # XXX: set layer naming convention explicitly? 
    #      Otherwise, names depend upon when model was created...
    model = cnn_model(img_rows=32, img_cols=32, channels=num_channels, nb_classes=num_classes)

    return model, x, y



def run_in_batches(sess, x_tf, y_tf, output_tf, x_in, y_in, batch_size):
    """ 
     Runs data through a CNN one batch at a time; gathers all results
     together into a single tensor.

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
        for batch in range(nb_batches):
            # Note: last batch may be smaller than all others...
            start = batch * batch_size
            end = min(n_examples, start + batch_size)
            cur_batch_size = end - start

            feed_dict = {x_tf : x_in[start:end], y_tf : y_in[start:end]}
            output_i = sess.run(output_tf, feed_dict=feed_dict)

            # the slice is to avoid any extra stuff in last mini-batch,
            # which might not be entirely "full"
            output_i = output_i[:cur_batch_size]
            out.append(output_i)

    out = np.concatenate(out, axis=0)
    return out



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
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, model_output, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
        preds = run_in_batches(sess, x, y, model_output, X_test, Y_test, FLAGS.batch_size)
        acc2 = 1. * np.sum(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1)) / Y_test.shape[0]
        print('second check: ' + str(acc2))

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



def attack_lisa_cnn(sess, cnn_weight_file, y_target=None):
    """ Generates AE for the LISA-CNN.
        Assumes you have already run train_lisa_cnn() to train the network.
    """
    # Note: we load the version of the data *without* extra context
    X_train, Y_train, X_test, Y_test = data_lisa(with_context=False)

    # This is the model we will attack
    model, x, y = make_lisa_cnn(sess, FLAGS.batch_size, X_train.shape[1])
    model_output = model(x)

    saver = tf.train.Saver()
    saver.restore(sess, cnn_weight_file)

    # create attack
    # TODO: wrap in cleverhans Model object
    fgsm = FastGradientMethod(model, sess=sess)

    if np.isscalar(y_target):
        # targeted attack
        y_vec = y_target * np.ones((FLAGS.batch_size,), dtype=np.int32)
        y_target_oh = to_categorical(y_vec, Y_test.shape[1])
        x_adv = fgsm.generate(x, eps=FLAGS.epsilon, y_target=y_target_oh)
    else:
        # non-targeted attack
        x_adv = fgsm.generate(x, eps=FLAGS.epsilon)

    #
    # Run the attack (targeted or untargeted)
    # on the test data.
    #
    if np.isscalar(y_target):
        tmp = y_target * np.ones((Y_test.shape[0],), dtype=np.int32)
        y_synthetic = to_categorical(tmp, Y_test.shape[1])
        X_adv = run_in_batches(sess, x, y, x_adv, X_test, y_synthetic, FLAGS.batch_size)
    else:
        X_adv = run_in_batches(sess, x, y, x_adv, X_test, Y_test, FLAGS.batch_size)

    #
    # Evaluate the AE. 
    # Here, we use the same model we attacked.
    #
    model_eval = model
    preds_tf = model_eval(x)
    preds = run_in_batches(sess, x, y, preds_tf, X_adv, Y_test, FLAGS.batch_size)
    print('Test accuracy on adversarial examples: %0.3f' % calc_acc(Y_test, preds))
    print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))

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
