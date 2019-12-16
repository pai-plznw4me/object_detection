import unittest
import tensorflow as tf
import numpy as np
from iou import calculate_iou


class TestIou(unittest.TestCase):
    def setUp(self):
        self.flat_xyxy_anchor = np.asarray([0, 0, 10, 10,
                                            0, 0, 5, 5,
                                            10, 0, 40, 10,
                                            0, 10, 10, 40,
                                            0, 0, 40, 40, # 완전히 포함
                                            -10, -10, 10, 10,
                                            -10, 0, 40, 10,
                                            10, -10, 40, 10,
                                            0, -17, 32, 47
                                            ])
        self.flat_xyxy_anchor_tf = tf.constant(self.flat_xyxy_anchor)
        self.gt_bboxes = np.asarray([[5, 5, 30, 30,
                                      10, 10, 40, 40,
                                      47,  44, 427, 174]])
        self.gt_bboxes_tf= tf.constant(self.gt_bboxes)

        float_formatter = "{:.8f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})

        # answer
        answer = [[0.03571429, 0.],
                  [0.35555556, 1.],
                  [0.12121212, 0.],
                  [0.12121212, 0.]]
        self.np_answer = np.asarray(answer)

    def test_calculate_iou(self):
        iou_tf = calculate_iou(flat_xyxy_bboxes=self.flat_xyxy_anchor_tf,
                                  flat_xyxy_gt=self.gt_bboxes_tf)
        sess = tf.Session()
        iou = sess.run(iou_tf)
        print(iou)
        np.testing.assert_array_almost_equal(iou, self.np_answer)

    def test_sample(self):
        from tensorflow.python.keras.applications import ResNet50
        from tensorflow.python.keras.layers import Input
        from datasets import sample_dataset
        from anchors import generate_anchor
        from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d

        inputs = Input(shape=(None, None, 3), name='images')
        backbone_layer = ResNet50(weights='imagenet', input_shape=(None, None, 3), include_top=False)(inputs)

        # load sample image , label
        sample_tensor, xyxy_gt = sample_dataset()

        # ground truth
        xyxy_gt = tf.constant(xyxy_gt, dtype=tf.float64)
        xyxy_flat_gt = tf.reshape(xyxy_gt, [-1])

        # generate anchors
        ccwh_anchors = generate_anchor(inputs, backbone_layer)
        ccwh_res_anchors = tf.reshape(ccwh_anchors, (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], 9, 4))
        xyxy_anchors = convert_ccwh4d_to_xyxy4d(ccwh_res_anchors)
        xyxy_flat_anchors = tf.reshape(xyxy_anchors, [-1])

        # iou
        iou_matrix = calculate_iou(xyxy_flat_anchors, xyxy_flat_gt)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        a = sess.run([xyxy_flat_gt, xyxy_flat_anchors, iou_matrix], {inputs: sample_tensor})
        print(a[1][:8])
        print(a[2][2])

