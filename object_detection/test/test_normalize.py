import unittest
import numpy as np
import tensorflow as tf
from normalize import normalize_anchors


class TestNormalize(unittest.TestCase):
    """
    Test 해야 할 것들.
    Anchor 가 지정한 위치에 잘 생성 되는지

    """
    def setUp(self):
        # Test Normalization
        self.anchors = [0, 0, 5, 6,
                        5, 5, 10, 11,
                        0, 0, 7, 6]
        self.anchors = np.asarray(self.anchors, dtype=np.float32)

        # gt 가 하나 있을때
        self.gt1 = [3, 3, 8, 8, 4, 4, 8, 8]
        self.gt1 = np.asarray(self.gt1, dtype=np.float32)

        answer1 = [[[0.6,         0.5,         0.47000363,  0.2876821],
                   [0.8,         0.6666667,   0.47000363,  0.2876821]],

                  [[-0.2, -0.18181819, -0.22314355, -0.31845373],
                   [-0.1, -0.09090909, -0.22314355, -0.31845373]],

                  [[0.42857143,  0.5,         0.13353139,  0.2876821],
                   [0.5714286,   0.6666667,   0.13353139,  0.2876821]]]
        self.answer1 = np.asarray(answer1)
        # gt 가 복수개 있을때

        # gt 가 하나 있을때
        self.gt2 = [3, 3, 8, 8]
        self.gt2 = np.asarray(self.gt2, dtype=np.float32)

        answer2 = [[[0.6,         0.5,         0.47000363,  0.2876821]],
                  [[-0.2, -0.18181819, -0.22314355, -0.31845373]],
                  [[0.42857143,  0.5,         0.13353139,  0.2876821]]]
        self.answer2 = np.asarray(answer2)

    def test_normalize_anchors(self):
        #
        norm_anchors1 = normalize_anchors(
            tf.constant(self.anchors), tf.constant(self.gt1))
        #
        norm_anchors2 = normalize_anchors(
            tf.constant(self.anchors), tf.constant(self.gt2))

        sess = tf.Session()
        norm_anchors_1 = sess.run(norm_anchors1)
        norm_anchors_2 = sess.run(norm_anchors2)

        np.testing.assert_array_almost_equal(norm_anchors_1, self.answer1)
        np.testing.assert_array_almost_equal(norm_anchors_2, self.answer2)
