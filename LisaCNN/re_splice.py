#!/usr/bin/env python
import pdb
import os
import tensorflow as tf
import cv2
import csv
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import cnn_model
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'tmp', 'Directory storing the saved model.')
flags.DEFINE_string('data_dir','/home/fendlnm1/Fendley/street_signs/cropped_images/', 'The Directory in which the extra lisadataset is')

flags.DEFINE_string('csv','/home/fendlnm1/Fendley/street_signs/signDatabase/allAnnotations.csv', 'The Directory in which the extra lisadataset is')
flags.DEFINE_string('data_full_dir','/home/fendlnm1/Fendley/street_signs/full_images/', 'The Directory in which the extra lisadataset is')
flags.DEFINE_string(
    'filename', 'lisacnn.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 60, 'Number of epochs to train model')
flags.DEFINE_integer('nb_classes', 48, 'Number of classes')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')


def main(argv=None):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1246)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Defined TensorFlow model graph.")
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))
    model = cnn_model(img_rows=32, img_cols=32, channels=3, nb_classes=FLAGS.nb_classes)
    predictions = model(x)
    fgsm_params = {'eps': float(1)/255}
    fgsm = FastGradientMethod(model, sess=sess)
   
    saver = tf.train.Saver()
   
    save_string = os.path.join(FLAGS.train_dir,FLAGS.filename)
    saver.restore(sess, save_string)
    model_2 = model
    predictions_2 = model_2(x)
    adv_x_2 = fgsm.generate(x,**fgsm_params)
    predictions_2_adv = model_2(adv_x_2)

    pic_list = os.listdir(FLAGS.data_dir)
    x_adv_out = []
    anno = {}
    x_crop = []
    cord_list = []
    with open(FLAGS.csv) as csvdata:
        csvread = csv.reader(csvdata)
        for row in csvread:
            data = row[0].split(';')
            if data[0] != 'Filename':
                filename = os.path.basename(data[0])
                cords = [int(data[2]), int(data[3]),int(data[4]), int(data[5])]
                anno[filename] = cords

    x_full_list = []
    max_cutoff = 1000
    pic_list = pic_list[:1000]

    for img_filename in pic_list:
        x_small = cv2.imread(os.path.join(FLAGS.data_dir, img_filename))
        full_filename = "_".join(img_filename.split('_')[1:])
        x_full = cv2.imread(os.path.join(FLAGS.data_full_dir, full_filename))
        x_full_list.append(x_full)
        x_resize = cv2.resize(x_small.copy(), (32, 32))
	x_adv_out.append(x_resize)
        x_crop.append(x_small)
        cords = anno[full_filename]
        cord_list.append(cords)
        

    x_adv = adv_x_2.eval(session=sess,feed_dict={x:x_adv_out})
    final = []
    for i in range(len(x_crop)):
        resize = cv2.resize(x_adv[i].copy(), (x_crop[i].shape[0],x_crop[i].shape[1]))
        output = x_full_list[i]
        output[cord_list[i][3]:cord_list[i][0]][cord_list[i][1]:cord_list[i][2]] = resize
        final.append(output)
    
    #adv_out = adv_x_2.eval(session=sess,feed_dict={x:X_crop})
    adv_out = np.asarray(final)
    adv_out *= 255
    counter = 0
    for img in adv_out:
        cv2.imwrite(os.path.join(FLAGS.train_dir,'adversarial_image'+str(counter)+'.jpg'), img)
        counter += 1



if __name__ == '__main__':
    app.run()
