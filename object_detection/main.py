from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.models import Model
import tensorflow as tf
from anchors import generate_anchor, generate_trainable_anchors
from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d
from normalize import normalize_anchors, denormalize
from iou import calculate_iou
from datasets import sample_dataset
from matching_policy import matching_mask_tf
from utils import mean_iou

inputs = Input(shape=(None, None, 3), name='images')
backbone_layer = ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False)(inputs)

# convolution for regression
n_anchor = 9
n_reg = 4
layer = Conv2D(filters = 128, activation='relu', kernel_size=3, padding='same')(backbone_layer)
layer = Conv2D(filters = 256, activation='relu', kernel_size=3, padding='same')(layer)
layer = Conv2D(filters = n_anchor * n_reg , activation='linear', kernel_size=3, padding='same')(layer)

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
best_matching_mask = matching_mask_tf(iou_matrix)

# trainable anchors
trainable_anchors = generate_trainable_anchors(norm_anchors, best_matching_mask)
trainable_anchors_4d = tf.reshape(trainable_anchors,
                                  (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], 9, 4))

trainable_anchors_3d = tf.reshape(trainable_anchors,
                                  (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], -1))

mask = tf.where(trainable_anchors_3d == -1, tf.zeros_like(trainable_anchors_3d), tf.ones_like(trainable_anchors_3d))

masked_pred = tf.cast(layer, tf.float64) * mask
masked_anchor = trainable_anchors_3d * mask

# denormalization -> denormalization 하면 groundtruth 가 나와야 한다.
# denorm_4d = denormalize(trainable_anchors_4d, ccwh_res_anchors)

# keras 는 axis = 0 을 batch 로 본다 .그래서
trainable_anchors_3d_train = tf.expand_dims(masked_anchor, axis=0)
model = Model(inputs, masked_pred)
model.compile(optimizer='adam', loss='mean_squared_error')
results = model.fit(sample_tensor/255., trainable_anchors_3d_train, batch_size=16, epochs=20, steps_per_epoch=10)
