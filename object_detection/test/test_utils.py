import unittest
import tensorflow as tf
import numpy as np
from utils import convert_xyxy2d_to_ccwh2d, convert_ccwh4d_to_xyxy4d


class TestAnchors(unittest.TestCase):
    """
    Test 해야 할 것들.

    # shape : [H,W,6] 인 tensor 을 [2,2,3,2] 로 Reshape 을 했을때
     정상적으로 배열 되었는지 확인한다

    sample_tensor, xyxy_gt = sample_dataset()

    # ground truth
    xyxy_gt = tf.constant(xyxy_gt, dtype=tf.float64)
    xyxy_flat_gt = tf.reshape(xyxy_gt, [-1])
    ccwh_gt = bboxes_xyxy2cxcywh_tf(xyxy_gt)
    ccwh_flat_gt = tf.reshape(ccwh_gt, [-1])

    # generate anchors
    ccwh_anchors = generate_anchor(inputs, backbone_layer)
    ccwh_res_anchors = tf.reshape(ccwh_anchors, (tf.shape(backbone_layer)[1], tf.shape(backbone_layer)[2], 9, 4))
    ccwh_flat_anchors = tf.reshape(ccwh_res_anchors, [-1])
    xyxy_anchors = anchor_cxcywh2xyxy_tf(ccwh_res_anchors )
    xyxy_flat_anchors = tf.reshape(xyxy_anchors, [-1])


    #

    """
    def setUp(self):
        # shape: 2,2,6
        self.test_tensor = np.asarray([[[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]],
                                       [[31, 32, 33, 34, 35, 36], [41, 42, 43, 44, 45, 46]]],
                                      dtype=np.float32)

        # shape: 2, 2, 4                 x1  y1  x2  y2    x`1 y`1 x`2 y`2
        self.test_tensor2 = np.asarray([[[11, 12, 13, 14], [21, 22, 23, 24]],
                                        [[31, 32, 33, 34], [41, 42, 43, 44]]],
                                       dtype=np.float32)
        # shape: 4, 4
        self.test_tensor3 = np.asarray([[11, 12, 13, 14],
                                        [21, 22, 23, 24],
                                        [31, 32, 33, 34],
                                        [41, 42, 43, 44]],
                                       dtype=np.float32)

        # shape: 2, 2, 2, 4
        self.test_tensor4 = \
            [[[[11., 12., 13., 14.], [15., 16., 17., 18.]], [[21., 22., 23., 24.], [25., 26., 27., 28.]]],
             [[[31., 32., 33., 34.], [35., 36., 37., 38.]], [[41., 42., 43., 44.], [45., 46., 47., 48.]]]]
        #       x1   y1   x2   y2     x1   y1   x2   y2       x1   y1   x2   y2     x1   y1   x2   y2

    def test_extract_X(self):
        """
        Description:
          x1  y1  x2  y2  x3  y3    x1  y1  x2  y2  x3  y3
        [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]],
         [[31, 32, 33, 34, 35, 36], [41, 42, 43, 44, 45, 46]]])
         위 matrix 에서 X {x1, x2, x3} 만 추출하는 코드를 실험해 본다

        :return:
        """
        # # shape : [2,2,3,2]
        #                      x1  x2  y3    y1  y2  y3
        answer = np.asarray([[[11, 13, 15], [21, 23, 25]],
                             [[31, 33, 35], [41, 43, 45]]])
        pred = self.test_tensor.reshape((2, 2, 3, 2))
        np.testing.assert_array_almost_equal(answer, pred[:, :, :, 0])

    def test_flat(self):
        """
        Description:
            [[[x1, y1], [x2, y2]],
             [[x3, y3], [x4, y5]]] 을 flat 했을때
              x1, y1, x2, y2, x3, y3, x4, y4 가 되는지 확인
        :return:
        """
        #                    x1  y1  x2  y2 x`1  y`1 x`2 y`2 ...
        answer = np.asarray([11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44])
        flat_test_tensor = tf.reshape(self.test_tensor2, [-1])

        # open Session
        sess = tf.Session()
        pred = sess.run(flat_test_tensor)
        np.testing.assert_array_almost_equal(pred, answer)

    def test_convert_xyxy2d_to_ccwh2d(self):
        answer = [[11 + (13 - 11) / 2, 12 + (14 - 12) / 2, (13 - 11), (14 - 12)],
                  [21 + (23 - 21) / 2, 22 + (24 - 22) / 2, (23 - 21), (24 - 22)],
                  [31 + (33 - 31) / 2, 32 + (34 - 32) / 2, (33 - 31), (34 - 32)],
                  [41 + (43 - 41) / 2, 42 + (44 - 42) / 2, (43 - 41), (44 - 42)]]
        ccwh_2d = convert_xyxy2d_to_ccwh2d(self.test_tensor3)
        sess = tf.Session()
        pred = sess.run(ccwh_2d)
        np.testing.assert_array_almost_equal(answer, pred)

    def test_convert_ccwh4d_to_xyxy4d(self):
        answer = \
            [[[[11-13/2., 12-14/2., 11+13/2., 12+14/2.], [15-17/2., 16-18/2., 15+17/2., 16+18/2.]],
              [[21-23/2., 22-24/2., 21+23/2., 22+24/2.], [25-27/2., 26-28/2., 25+27/2., 26+28/2.]]],
             [[[31-33/2., 32-34/2., 31+33/2., 32+34/2.], [35-37/2., 36-38/2., 35+37/2., 36+38/2.]],
              [[41-43/2., 42-44/2., 41+43/2., 42+44/2.], [45-47/2., 46-48/2., 45+47/2., 46+48/2.]]]]
        self.test_tensor4 = tf.constant(self.test_tensor4)
        ccwh_4d = convert_ccwh4d_to_xyxy4d(self.test_tensor4)
        sess = tf.Session()
        pred = sess.run(ccwh_4d)
        print(pred)
        np.testing.assert_array_almost_equal(answer, pred)
