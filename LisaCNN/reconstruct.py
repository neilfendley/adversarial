#!/usr/bin/env python

""" Pastes adversarial images back into the original scenes.

   Example usage:

     python reconstruct.py `find ./output/Images/Iterative_FGM_0.15 -name \*png -print`
"""

import os, sys
from PIL import Image
import pdb

import numpy as np

import subimage


def read_all_images(dirname):
    pass



if __name__ == "__main__":
    annotation_file = '~/Data/LISA/allAnnotations.csv'
    data_file = './output_jan5/lisa_data.npz'

    file_names = sys.argv[2:]

    # load context; assumes images are from test data split!
    si = subimage.parse_LISA(annotation_file)
    f = np.load(data_file)
    test_indices = f['test_idx']

    # load all sub-images
    sub_images = []
    image_ids = []
    for fn in file_names:
        im = Image.open(fn)
        head, tail = os.path.split(fn)

        # deterime the image id in the *test set*
        pieces = tail.split('_')
        if pieces[0].startswith('image'):
            pieces[0] = pieces[0][5:]
        test_image_idx = int(pieces[0])

        # determine original image id in the full set
        full_image_id = test_indices[test_image_idx]

        # splice into original image
        full_img = si.splice_subimages([full_image_id,], [np.array(im),])[0]

        # save result (in same directory)
        out_fn, ext = os.path.splitext(fn)
        out_fn = out_fn + "_full_image" + ext
        img = Image.fromarray(full_img.astype('uint8'))
        img.save(out_fn)
