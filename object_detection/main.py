import sys
sys.path.append('./object_detection/object_detection')
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
import tensorflow as tf
from anchors import generate_anchor, generate_trainable_anchors
from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d
from normalize import normalize_anchors, denormalize
from iou import calculate_iou
from datasets import sample_dataset
from matching_policy import matching_mask
from utils import mean_iou
import tensorflow.python.keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt


inputs = Input(shape=(None, None, 3), name='images')
backbone_layer = ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False)(inputs)

# convolution for regression
n_anchor = 9
n_reg = 4
layer = Conv2D(filters=128, activation='relu', kernel_size=3, padding='same')(backbone_layer)
layer = Conv2D(filters=256, activation='relu', kernel_size=3, padding='same')(layer)
layer = Conv2D(filters=n_anchor * n_reg, activation='linear', kernel_size=3, padding='same')(layer)

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

# trainable anchors
trainable_anchors = generate_trainable_anchors(norm_anchors, best_matching_mask)
trainable_anchors_4d = tf.reshape(trainable_anchors,
                                  (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], 9, 4))
trainable_anchors_3d = tf.reshape(trainable_anchors,
                                  (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], -1))

mask = tf.where(tf.equal(trainable_anchors_3d, -1),
                tf.zeros_like(trainable_anchors_3d), tf.ones_like(trainable_anchors_3d))

masked_pred = tf.cast(layer, tf.float64) * mask
masked_anchor = trainable_anchors_3d * mask

n_valid_anchors = tf.reduce_sum(mask, axis=None) / 4


def mse(masked_anchor, masked_pred):
    mask = tf.where(tf.greater_equal(masked_anchor, 0),
                    tf.ones_like(masked_anchor), tf.zeros_like(masked_anchor))
    n_pos_anchors = tf.reduce_sum(mask) / 4

    return tf.reduce_sum(tf.abs((masked_anchor - masked_pred))) / n_pos_anchors


# denormalization -> denormalization 하면 groundtruth 가 나와야 한다.
res_layer = tf.reshape(layer, [tf.shape(layer)[1], tf.shape(layer)[2], 9, 4])
res_layer = tf.cast(res_layer, tf.float64)
denorm_4d = denormalize(res_layer, ccwh_res_anchors)

# keras 는 axis = 0 을 batch 로 본다 .그래서
trainable_anchors_3d_train = tf.expand_dims(masked_anchor, axis=0)
model = Model(inputs, masked_pred)

sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd, loss=mse)

sess = K.get_session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    a = sess.run(denorm_4d, {inputs:sample_tensor/255})
    mask_ = sess.run(mask, {inputs:sample_tensor/255})
    res_mask_ = np.reshape(mask_ , [-1, 4])
    print(np.where(res_mask_[:, 0] !=0))
    print(a.shape)
    print(mask_.shape)

    res_a = a.reshape([-1, 4])
    print(res_a[485])
    x1=res_a[485][0] - res_a[485][2]/2
    y1=res_a[485][1] - res_a[485][3]/2
    x2=res_a[485][0] + res_a[485][2]/2
    y2=res_a[485][1] + res_a[485][3]/2
    [x1,y1,x2,y2] = map(int, [x1,y1,x2,y2])
    show_image = sample_tensor[0].copy()
    cv2.rectangle(show_image, (x1,y1), (x2,y2), (255,0,0), 5)
    plt.imshow(show_image)
    plt.show()
    results = model.fit(sample_tensor/255., trainable_anchors_3d_train, batch_size=16, epochs=10, steps_per_epoch=100)