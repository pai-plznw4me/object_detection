import unittest
from normalize import normalize_anchors
from anchors import generate_trainable_anchors
import numpy as np
import tensorflow as tf


class TestAnchors(unittest.TestCase):
    """
    Test 해야 할 것들.
    Anchor 가 지정한 위치에 잘 생성 되는지
    1.
    """
    def setUp(self):
        # Test Normalization
        anchors = np.asarray([0, 0, 5, 6,
                                   5, 5, 10, 11,
                                   0, 0, 7, 6], dtype=np.float32)
        gt = np.asarray([3, 3, 8, 8, 4, 4, 8, 8], dtype=np.float32)

        self.pos_mask = np.asarray([[-1, 1],
                               [-1, -1],
                               [1, 1]])

        self.norm_anchors = normalize_anchors(anchors, gt)
        answer = ([[0.8, 0.6666667, 0.47000363, 0.2876821],
                    [-1., -1., -1., -1.],
                    [0.5714286, 0.6666667, 0.13353139, 0.2876821]])
        self.answer = np.asarray(answer)

    def test_generate_anchor(self):
        trainable_anchors = generate_trainable_anchors(self.norm_anchors, self.pos_mask)
        np.testing.assert_array_almost_equal(self.answer, trainable_anchors)

    def test_mask(self):
        indices_2d = tf.where(self.pos_mask == 1)
        indices_2d = tf.stack(indices_2d, axis=0)
        indices = indices_2d[:, 0]
        #
        sess = tf.Session()
        indices_ = sess.run(indices)

    def test_delta_norm_tf(self):
        """
        Description:

        :return:
        """

        # Numpy
        indices_2d = np.where(self.pos_mask  == 1)
        indices_2d = np.stack(indices_2d, axis=0).tolist()

        # trainable axis=0 기준으로 어디에다가 추출한 x,y,w,h 을 넣어야 할지 가리키는 indices
        indices = indices_2d[0]

        # delta 에서 해당 좌표를 가져온다
        dx = self.norm_anchors[:, :, 0][indices_2d]
        dy = self.norm_anchors[:, :, 1][indices_2d]
        dw = self.norm_anchors[:, :, 2][indices_2d]
        dh = self.norm_anchors[:, :, 3][indices_2d]
        print(dx)
        answer = np.stack([dx, dy, dw, dh], axis=-1)

        # Tensorflow
        # indexing
        indices_2d = tf.where(self.pos_mask == 1)
        print(indices_2d)
        indices_2d = tf.stack(indices_2d, axis=0)
        indices = indices_2d[:,0]
        indices = tf.expand_dims(indices, axis=-1)

        # calculate delta
        dx = tf.gather_nd(self.norm_anchors[:, :, 0], [indices_2d])
        dy = tf.gather_nd(self.norm_anchors[:, :, 1], [indices_2d])
        dw = tf.gather_nd(self.norm_anchors[:, :, 2], [indices_2d])
        dh = tf.gather_nd(self.norm_anchors[:, :, 3], [indices_2d])
        d_xywh = tf.stack([dx, dy, dw, dh], axis=-1)

        sess = tf.Session()
        d_xywh_= sess.run(d_xywh)[0]

        np.testing.assert_array_almost_equal(answer, d_xywh_)

    def test_generate_anchor_tf(self):
        answer = generate_trainable_anchors(self.norm_anchors, self.pos_mask)
        trainable_anchors = generate_trainable_anchors(self.norm_anchors, self.pos_mask)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        trainable_anchors_ = sess.run(trainable_anchors)

        np.testing.assert_array_almost_equal(answer, trainable_anchors_)




