"""
Code to facilitate working with sub-images/regions within larger images.

This module also contains some LISA-specific codes (that could be moved elsewhere later).
"""

__author__ = "mjp"
__date__ = "dec 2017"


import os
import pdb

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split


class Subimage(object):
  def __init__(self, to_grayscale=False):
    self._filenames = []
    self._bbox = []
    self._y = []
    self._to_grayscale = to_grayscale


  def __str__(self):
    out = '%d images' % len(self._filenames)
    out += ' with %d unique classes' % len(np.unique(np.array(self._y)))
    return out


  def add(self, filename, bbox, y):
    self._filenames.append(filename)
    self._bbox.append(bbox)
    self._y.append(y)


  def describe(self, indices):
    y = np.array(self._y)
    y = y[indices]
    out = '%d objects with %d unique class labels\n' % (len(indices), np.unique(y).size)
    for yi in np.unique(y):
      out += '  y=%d : %d\n' % (yi, np.sum(y == yi))
    return out


  def train_test_split(self, pct_test, max_per_class=np.Inf, reserve_for_test=[]):
    assert(pct_test < 1.0)

    # (optional): there may be certain images we want to explicitly reserve for test
    reserved_indices = []
    for pattern in reserve_for_test:
      for idx, filename in enumerate(self._filenames):
        if pattern in filename:
          reserved_indices.append(idx)
    reserved_indices = np.array(reserved_indices, dtype=np.int32)

    # generate the train/test split
    indices = np.delete(np.arange(len(self._y)), reserved_indices)
    class_labels = np.delete(np.array(self._y, dtype=np.int32), reserved_indices)

    train, test = train_test_split(indices, test_size=pct_test, stratify=class_labels)
    test = np.concatenate([test, np.array(reserved_indices, dtype=np.int32)])

    # (optional): limit max # of examples in a given class (for training)
    if np.isfinite(max_per_class):
      y = np.array(self._y)

      for yi in np.unique(y):
        yi_idx = np.nonzero(y[train] == yi)[0]
        if len(yi_idx) > max_per_class:
          train = np.delete(train, yi_idx[max_per_class:])

    return train, test


  def get_images(self, indices):
    "Returns full sized images at the specified indices."
    out = []
    for idx in indices:
      im = Image.open(self._filenames[idx])

      if self._to_grayscale:
        im = im.convert('L')

      out.append(np.array(im))

    y = np.array(self._y)
    return out, y[indices]


  def get_subimages(self, indices, new_size=None, pct_context=0):
    "Extracts sub-indices from images."
    out = []
    for idx in indices:
      im = Image.open(self._filenames[idx])
      if self._to_grayscale:
        im = im.convert('L')

      bbox = self._bbox[idx]

      # (optional) expand box to grab additional context.
      # currently assumes sub-images are well within the interior of the image.
      if pct_context > 0:
        dx = bbox[2] - bbox[0]
        dy = bbox[3] - bbox[1]
        dx_new = dx * (1. + pct_context)
        dy_new = dy * (1. + pct_context)

        x0 = np.floor((bbox[2] + bbox[0])/2 - dx_new/2.)
        x1 = np.floor((bbox[2] + bbox[0])/2 + dx_new/2.)
        y0 = np.floor((bbox[3] + bbox[1])/2 - dy_new/2.)
        y1 = np.floor((bbox[3] + bbox[1])/2 + dy_new/2.)

        bbox = [int(x0), int(y0), int(x1), int(y1)]

      # crop out the sub-image
      im = im.crop(bbox)

      # (optional) resize subimage
      if new_size is not None:
        im = im.resize(new_size, Image.ANTIALIAS)

      im_arr = np.array(im)
      if im_arr.ndim == 2:
        im_arr = im_arr[:,:,np.newaxis] # force channel dimension

      out.append(np.array(im_arr))

    y = np.array(self._y)
    return out, y[indices]


  def splice_subimages(self, indices, new_subimage):
    """
       Splice new subimages into original images.

         new_subimage : either (a) a list of images to inject or 
                               (b) a tensor of images (n x rows x cols x channels)
    """
    out = []

    for ii, idx in enumerate(indices):
      # extract the ith sub-image
      if isinstance(new_subimage, list) or isinstance(new_subimage, tuple):
        si = new_subimage[ii]
      else:
        si = new_subimage[ii,...]
      assert(si.ndim >= 2)

      # corresponding full scene
      im = Image.open(self._filenames[idx])
      if self._to_grayscale:
        im = im.convert('L')
      xi = np.array(im)

      # the bounding box
      x0,y0,x1,y1 = self._bbox[idx]
      width = x1-x0
      height = y1-y0

      # resize new image (if needed) and blast into bounding box
      if si.shape[0] != height or si.shape[1] != width:
        si = Image.fromarray(si).resize((width,height), Image.ANTIALIAS)
        si = np.array(si)

      xi[y0:y1,x0:x1] = np.squeeze(si)
      out.append(xi)

    return out

#-------------------------------------------------------------------------------

#LISA_CLASS_MAP = {"rightLaneMustTurn": 8, "thruMergeRight": 45, "yieldAhead": 23, "turnRight": 18, "pedestrian": 17, "signalAhead": 9, "thruMergeLeft": 40, "slow": 3, "noLeftTurn": 24, "merge": 5, "turnLeft": 35, "noRightTurn": 11, "rampSpeedAdvisory": 22, "zoneAhead45": 7, "rampSpeedAdvisory20": 33, "doNotEnter": 46, "speedLimit30": 12, "exitSpeedAdvisory": 42, "speedLimit35": 0, "noUTurn": 25, "stop": 2, "stopAhead": 14, "curveLeft": 41, "dip": 21, "leftTurn": 28, "roundabout": 26, "schoolSpeedLimit25": 30, "speedLimit": 1, "intersection": 36, "truckSpeedLimit55": 44, "keepRight": 6, "truckSpeedLimit": 37, "school": 16, "rightLaneEnds": 38, "schoolSpeedLimit": 29, "pedestrianCrossing": 10, "curve": 27, "zoneAhead25": 34, "yield": 15, "curveRight": 39, "addedLane": 4, "rampSpeedAdvisory35": 31, "laneEnds": 13, "speedLimit25": 19, "speedLimit65": 32, "speedLimitWhenFlashing": 43, "doNotPass": 47, "speedLimit45": 20}

# These correspond to table 1 from Evtimov et al. "Robust Physical-World Attacks on Deep Learning Models".
# It does *not* capture all possible classes in LISA.
#
LISA_17_CLASSES = ["addedLane", "keepRight", "laneEnds", "merge", "pedestrianCrossing", "school",
                   "schoolSpeedLimit25", "signalAhead", "speedLimit25", "speedLimit30", "speedLimit35",
                   "speedLimit45", "speedLimitUrdbl", "stop", "stopAhead", "turnRight", "yield"]

LISA_17_CLASS_MAP = { x : ii for ii,x in enumerate(LISA_17_CLASSES) }


def parse_LISA(csvfile, class_map=LISA_17_CLASS_MAP):
  "See also: tools/extractAnnotations.py in LISA dataset."
  si = Subimage(to_grayscale=True)

  csvfile = os.path.expanduser(csvfile)

  csv = open(csvfile, 'r')
  csv.readline() # discard header
  csv = csv.readlines()

  # Note: the original LISA parsing code shuffled the rows, but this just adds 
  #       potential confusion so I'm not doing that for now.

  # If no classmap was provided, create one.
  if class_map is None:
    class_map = {}
    for line in csv:
      fields = line.split(';')
      if fields[1] not in class_map:
        class_map[fields[1]] = len(class_map)
  
  # base path to actual filenames.
  base_path = os.path.dirname(csvfile)

  # do it
  for idx, line in enumerate(csv):
    fields = line.split(';')
    y_str = fields[1]
    if y_str not in class_map:
      continue

    im_filename = fields[0]
    y = class_map[y_str]
    x0 = int(fields[2])
    x1 = int(fields[4])
    y0 = int(fields[3])
    y1 = int(fields[5])
    bbox = [x0,y0,x1,y1]

    si.add(os.path.join(base_path, im_filename), bbox, y)

  return si

#-------------------------------------------------------------------------------


def _test_splicing():
  si = parse_LISA('~/Data/LISA/allAnnotations.csv')

  # ensures that, without resizing, one can paste the
  # exact sub-image back into the original image
  indices = np.arange(100)
  x_big, _ = si.get_images(indices)
  x_small, _ = si.get_subimages(indices, None)
  x_splice = si.splice_subimages(indices, x_small)

  for ii in range(len(x_splice)):
    assert(np.all(x_splice[ii] == x_big[ii]))

  return si


def _test_reserve_images():
  si = parse_LISA('~/Data/LISA/allAnnotations.csv')
  prefix = 'stop_1323803184.avi_image'
  train, test = si.train_test_split(.17, max_per_class=500, reserve_for_test=[prefix,])

  for idx in train:
    assert(not prefix in si._filenames[idx])



if __name__ == "__main__":
  _test_splicing() 
  _test_reserve_images()

  si = parse_LISA('~/Data/LISA/allAnnotations.csv')

  # this should approximate table I in Evtimov et al. fairly closely
  train_idx, test_idx = si.train_test_split(.17, max_per_class=500)
  print(si.describe(train_idx))
  print(si.describe(test_idx))

  # save sub-images to file for manual inspection
  print('extracting sub-images...')
  x_test, y_test = si.get_subimages(test_idx, (32,32), pct_context=.5)
  x_test = np.array(x_test) # [] -> tensor
  print(x_test.shape) # TEMP
  np.savez('test_images.npz', x_test=x_test, y_test=y_test)
