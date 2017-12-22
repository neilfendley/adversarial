"""
Code to facilitate working with sub-images/regions within larger images.
"""

__author__ = "mjp"
__date__ = "dec 2017"


import os
import pdb

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split


class Subimage(object):
  def __init__(self):
    self._filenames = []
    self._bbox = []
    self._y = []


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


  def train_test_split(self, pct_test, max_per_class=np.Inf):
    indices = np.arange(len(self._filenames))
    assert(pct_test < 1.0)

    train, test = train_test_split(indices, test_size=pct_test, stratify=self._y)

    # (optional): limit max # of examples in a given class
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
      out.append(np.array(im))

    y = np.array(self._y)
    return out, y[indices]


  def get_subimages(self, indices, new_size=None):
    "Extracts sub-indices from images."
    out = []
    for idx in indices:
      im = Image.open(self._filenames[idx])
      im = im.crop(self._bbox[idx])
      if new_size is not None:
        im = im.resize(new_size, Image.ANTIALIAS)
      out.append(np.array(im))

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
      xi = np.array(Image.open(self._filenames[idx]))

      # the bounding box
      x0,y0,x1,y1 = self._bbox[idx]
      width = x1-x0
      height = y1-y0

      # resize new image (if needed) and blast into bounding box
      if isinstance(new_subimage, list) or isinstance(new_subimage, tuple):
        si = new_subimage[ii]
      else:
        si = new_subimage[ii,...]
      assert(si.ndim >= 2)

      if si.shape[0] != height or si.shape[1] != width:
        si = Image(xi).resize((width,height), Image.ANTIALIAS)
        si = np.array(xi)

      xi[y0:y1,x0:x1] = si
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
  si = Subimage()

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

if __name__ == "__main__":
  # example usage
  si = parse_LISA('~/Data/LISA/allAnnotations.csv')

  # this should approximate table I in Evtimov et al. fairly closely
  train_idx, test_idx = si.train_test_split(.17, max_per_class=500)

  print(si.describe(train_idx))
  print(si.describe(test_idx))

  print('extracting sub-images...')
  x_test, y_test = si.get_subimages(test_idx, (32,32))
  x_test = np.array(x_test) # condense into a tensor
  print('done!')

  # sanity check the splicing operation
  print('testing splice code...')
  indices = test_idx[:100]
  x_big, _ = si.get_images(indices)
  x_small, _ = si.get_subimages(indices, None)
  x_splice = si.splice_subimages(indices, x_small)

  for ii in range(len(x_splice)):
    assert(np.all(x_splice[ii] == x_big[ii]))
