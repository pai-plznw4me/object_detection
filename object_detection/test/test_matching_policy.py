import unittest
from matching_policy import matching_mask, matching_mask_tf
import numpy as np
import numpy.random as npr
import tensorflow as tf


class TestMatchingPolicy(unittest.TestCase):
    """
    Test 해야 할 것들.
    Anchor 가 지정한 위치에 잘 생성 되는지
    1.
    """
    def setUp(self):
        np.set_printoptions(formatter={'float_kind': "{:.2f}".format})
        npr.seed(1)

        n_row = 10
        n_gt = 2

        answer = [[-1,  1],
                  [-1, -1],
                  [-1, -1],
                  [-1, -1],
                  [-1, -1],
                  [ 1, -1],
                  [-1,  1],
                  [-1, -1],
                  [-1, -1],
                  [-1, -1]]

        n_classes = [1, 2]
        onehot = [[0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]

        self.answer = np.asarray(answer, dtype=np.float32)
        self.sample_anchors = np.arange(n_row * 4)
        self.anchor_flag = np.ones([n_row]) * -1
        self.sample_iou = npr.rand(n_row, n_gt)

    def test_matching_mask(self):
        self.matcing_mask = matching_mask(self.sample_iou)
        np.testing.assert_array_almost_equal(self.answer, self.matcing_mask)

    def test_matching_mask_tf(self):
        self.matcing_mask_tensor = matching_mask_tf(tf.constant(self.sample_iou, dtype=tf.float32))
        # best_mask, thd_mask = matching_mask_tf(tf.constant(self.sample_iou, dtype=tf.float32))
        sess = tf.Session()
        matcing_mask = sess.run(self.matcing_mask_tensor)
        # best_mask_, thd_mask_ = sess.run([best_mask, thd_mask])
        # print(best_mask_, thd_mask)
        np.testing.assert_array_almost_equal(matcing_mask, self.answer)

