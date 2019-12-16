from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input
import tensorflow as tf
from anchors import generate_anchor, generate_trainable_anchors
from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d
from normalize import normalize_anchors, denormalize
from iou import calculate_iou
from datasets import sample_dataset
from matching_policy import matching_mask_tf

inputs = Input(shape=(None, None, 3), name='images')
backbone_layer = ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False)(inputs)

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
# denormalization -> denormalization 하면 groundtruth 가 나와야 한다.
denorm_4d = denormalize(trainable_anchors_4d, ccwh_res_anchors)



"""
# # tmp
# tmp = tf.equal(best_matching_mask, 1)
# indices_2d = tf.where(tmp)
# print(indices_2d)
# # indices_2d = tf.stack(indices_2d, axis=0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(trainable_anchors, {inputs: sample_tensor})
denorm_4d_ = sess.run(denorm_4d, {inputs: sample_tensor})
denorm_4d_ = denorm_4d_.reshape([-1, 4])
import numpy as np
ind = np.where(a[:,0] != -1)[0]
target_coordinate = denorm_4d_[ind]
print(target_coordinate)

target_coordinate = target_coordinate[0]
x = target_coordinate[0] - target_coordinate[2]/2
y = target_coordinate[1] - target_coordinate[3]/2
x2 = target_coordinate[0] + target_coordinate[2]/2
y2 = target_coordinate[1] + target_coordinate[3]/2
print(x,y,x2,y2)
[x,y,x2,y2] = map(int, [x,y,x2,y2])
import matplotlib.pyplot as plt
import cv2
sample_tensor, xyxy_gt = sample_dataset()
a = sample_tensor[0]
gt_x, gt_y, gt_x2, gt_y2 = xyxy_gt[0]
cv2.rectangle(a, (x,y),(x2,y2),(255,0,0),5)
cv2.rectangle(a, (gt_x, gt_y), (gt_x2, gt_y2), (0, 255,0), 5)
print(gt_x, gt_y, gt_x2, gt_y2 )
plt.imshow(a)
plt.show()
"""


"""
    # Tensorflow
    indices_2d = tf.where(tf.equal(matching_mask, 1))
    indices_2d = tf.stack(indices_2d, axis=0)

    indices = indices_2d[:, 0]
    indices = tf.expand_dims(indices, axis=-1)

    # calculate delta
    # [0] 을 붙이는 이유는 tf.gather_nd 을 사용하고 나면 출력 tensor의 shape 가 (1, N, 4) 로 나온다
    # 1 은 필요없어 제거하기 위해 [0]을 붙인다
    dx = tf.gather_nd(normalize_anchors[:, :, 0], [indices_2d])[0]
    dy = tf.gather_nd(normalize_anchors[:, :, 1], [indices_2d])[0]
    dw = tf.gather_nd(normalize_anchors[:, :, 2], [indices_2d])[0]
    dh = tf.gather_nd(normalize_anchors[:, :, 3], [indices_2d])[0]

    d_xywh = tf.stack([dx, dy, dw, dh], axis=-1)

    ret_anchor = tf.ones([len(normalize_anchors), 4], dtype=tf.float32) * -1
    ret_anchor = tf.tensor_scatter_nd_update(ret_anchor, indices, d_xywh)
    return ret_anchor
"""