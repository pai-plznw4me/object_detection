import sys

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
import tensorflow as tf
from anchors import generate_anchor, generate_trainable_anchors, generate_trainble_classes
from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d
from normalize import normalize_anchors, denormalize
from iou import calculate_iou
from datasets import sample_dataset
from matching_policy import matching_mask
from utils import mean_iou

inputs = Input(shape=(None, None, 3), name='images')
backbone_layer = ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False)(inputs)

# convolution for regression
n_anchor = 9
n_classes = 1+1
layer = Conv2D(filters = 128, activation='relu', kernel_size=3, padding='same')(backbone_layer)
layer = Conv2D(filters = 256, activation='relu', kernel_size=3, padding='same')(layer)
layer = Conv2D(filters = n_anchor * n_classes , activation='softmax', kernel_size=3, padding='same')(layer)

# load sample image , label
sample_tensor, xyxy_gt = sample_dataset()

# ground truth
xyxy_gt = tf.constant(xyxy_gt, dtype=tf.float64)
xyxy_flat_gt = tf.reshape(xyxy_gt, [-1])
ccwh_gt = convert_xyxy2d_to_ccwh2d(xyxy_gt)
ccwh_flat_gt = tf.reshape(ccwh_gt, [-1])

# generate anchors
ccwh_anchors = generate_anchor(inputs, backbone_layer)
ccwh_res_anchors = tf.reshape(ccwh_anchors, (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], 9, 4))
ccwh_flat_anchors = tf.reshape(ccwh_res_anchors, [-1])
xyxy_anchors = convert_ccwh4d_to_xyxy4d(ccwh_res_anchors)
xyxy_flat_anchors = tf.reshape(xyxy_anchors, [-1])

# iou
iou_matrix = calculate_iou(xyxy_flat_anchors, xyxy_flat_gt)

# normalize
norm_anchors = normalize_anchors(ccwh_flat_anchors, ccwh_flat_gt)

# mask
best_matching_mask = matching_mask(iou_matrix)

# trainble classes
trainable_classes = generate_trainble_classes(tf.cast(best_matching_mask, dtype=tf.int64),
                                              tf.constant([1], dtype=tf.int64),
                                              n_classes)
res_trainable_classes = tf.reshape(trainable_classes,
               (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], -1))

#
model = Model(inputs, layer)

#
sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
results = model.fit(sample_tensor/255., res_trainable_classes,
                    batch_size=16, epochs=10, steps_per_epoch=100)
