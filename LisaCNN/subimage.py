"""
Code to facilitate working with sub-images/regions within larger images.
"""

__author__ = "mjp"


import os
import numpy as np
from PIL import Image
import pdb


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


  def sample(self, max_per_class=np.Inf):
    indices = []
    return indices


  def get_subimages(self, indices, new_size):
    out = []
    for idx in indices:
      im = Image.open(self._filenames[idx])
      im = im.crop(self._bbox[idx])
      im = im.resize(new_size, Image.ANTIALIAS)
      out.append(np.array(im))
    return np.array(out)


  def splice(self, indices, new_subimage):
    pass


#-------------------------------------------------------------------------------

#LISA_CLASS_MAP = {"rightLaneMustTurn": 8, "thruMergeRight": 45, "yieldAhead": 23, "turnRight": 18, "pedestrian": 17, "signalAhead": 9, "thruMergeLeft": 40, "slow": 3, "noLeftTurn": 24, "merge": 5, "turnLeft": 35, "noRightTurn": 11, "rampSpeedAdvisory": 22, "zoneAhead45": 7, "rampSpeedAdvisory20": 33, "doNotEnter": 46, "speedLimit30": 12, "exitSpeedAdvisory": 42, "speedLimit35": 0, "noUTurn": 25, "stop": 2, "stopAhead": 14, "curveLeft": 41, "dip": 21, "leftTurn": 28, "roundabout": 26, "schoolSpeedLimit25": 30, "speedLimit": 1, "intersection": 36, "truckSpeedLimit55": 44, "keepRight": 6, "truckSpeedLimit": 37, "school": 16, "rightLaneEnds": 38, "schoolSpeedLimit": 29, "pedestrianCrossing": 10, "curve": 27, "zoneAhead25": 34, "yield": 15, "curveRight": 39, "addedLane": 4, "rampSpeedAdvisory35": 31, "laneEnds": 13, "speedLimit25": 19, "speedLimit65": 32, "speedLimitWhenFlashing": 43, "doNotPass": 47, "speedLimit45": 20}

# These correspond to table 1 from Evtimov et al. "Robust Physical-World Attacks on Deep Learning Models".
# It does *not* capture all possible classes in LISA.
#
LISA_CLASSES = ["addedLane", "keepRight", "laneEnds", "merge", "pedestrianCrossing", "school",
                "schoolSpeedLimit25", "signalAhead", "speedLimit25", "speedLimit30", "speedLimit35",
                "speedLimit45", "speedLimitUrdbl", "stop", "stopAhead", "turnRight", "yield"]
LISA_CLASS_MAP = { x : ii for ii,x in enumerate(LISA_CLASSES)}


def parse_LISA(csvfile, class_map=LISA_CLASS_MAP):
  "See also: tools/extractAnnotations.py in LISA dataset."
  si = Subimage()

  csvfile = os.path.expanduser(csvfile)

  csv = open(csvfile, 'r')
  csv.readline() # discard header
  
  # base path to actual filenames.
  base_path = os.path.dirname(csvfile)

  # do it
  for idx, line in enumerate(csv.readlines()):
    fields = line.split(';')
    y_str = fields[1]
    if y_str not in LISA_CLASS_MAP:
      continue

    im_filename = fields[0]
    y = LISA_CLASS_MAP[y_str]
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
  x = si.get_subimages([1,3,23,50], (32,32))


