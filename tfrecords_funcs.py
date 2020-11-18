# Written by Dr Daniel Buscombe, Marda Science LLC
# for "ML Mondays", a course supported by the USGS Community for Data Integration
# and the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from data_imports import *

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np
import tensorflow.keras.backend as K

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

###############################################################
### TFRECORD FUNCTIONS
###############################################################

@tf.autograph.experimental.do_not_convert
def read_seg_tfrecord_4ddunes(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_png(example['image'], channels=4)
    image = tf.cast(image, tf.float32)/ 255.0

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)

    cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*11) #8)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*11, label)

    label = tf.one_hot(tf.cast(label, tf.uint8),12)
    label = tf.squeeze(label)

    return image, label

#-----------------------------------
@tf.autograph.experimental.do_not_convert
def read_seg_tfrecord_dunes(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)

    cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*11)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*11, label)

    label = tf.one_hot(tf.cast(label, tf.uint8), 12)
    label = tf.squeeze(label)

    return image, label

#-----------------------------------
def read_seg_image_and_label_dunes(img_path, flag='3d'):
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    dem_path = tf.strings.regex_replace(img_path, "images", "dems")
    dem_path = tf.strings.regex_replace(dem_path, "augimage", "augdem")
    bits = tf.io.read_file(dem_path)
    dem = tf.image.decode_jpeg(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    lab_path = tf.strings.regex_replace(img_path, "images", "labels")
    lab_path = tf.strings.regex_replace(lab_path, "augimage", "auglabel")
    lab_path = tf.strings.regex_replace(lab_path, ".jpg", "_label.jpg")
    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits, channels=1)

    label = tf.cast(label, tf.uint8)+1
    label = tf.squeeze(label,-1)
    print(label.shape)
    # label[image[:,:,0]==0] = 0

    cond = tf.equal(label, tf.zeros(tf.shape(label),dtype=tf.uint8))
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*11, label)

    cond = tf.equal(image[:,:,0], tf.zeros(tf.shape(label),dtype=tf.uint8))
    label = tf.where(cond, tf.zeros(tf.shape(label),dtype=tf.uint8), label)
    print(label.shape)

    ## merge dem and image
    if flag is '3d':
      merged = tf.stack([image[:,:,0], image[:,:,1], image[:,:,2]], axis=2)
    elif flag is '3db':
      merged = tf.stack([image[:,:,0], image[:,:,2], dem[:,:,0]], axis=2)
    else:
      merged = tf.stack([image[:,:,0], image[:,:,1], image[:,:,2], dem[:,:,0]], axis=2)

    merged = tf.cast(merged, tf.uint8)

    return merged, tf.expand_dims(label,-1)

#-----------------------------------
def recompress_seg_image4d(image, label):

    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_png(image)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_png(label)
    return image, label

#-----------------------------------
def recompress_seg_image(image, label):
    """
    "recompress_seg_image"
    This function takes an image and label encoded as a byte string
    and recodes as an 8-bit jpeg
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_jpeg(label, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

#-----------------------------------
def to_seg_tfrecord(img_bytes, label_bytes):
    """
    "to_seg_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label_bytes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "label": _bytestring_feature([label_bytes]), # one label image in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def seg_file2tensor(f):
    """
    "seg_file2tensor(f)"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_jpeg(bits)

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # image = tf.cast(image, tf.uint8) #/ 255.0

    return image
