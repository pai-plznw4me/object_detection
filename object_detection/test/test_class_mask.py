import sys
sys.path.append('../')
import unittest
import numpy as np
import tensorflow as tf
from anchors import generate_trainble_classes


class TestClassMask(unittest.TestCase):

    def setUp(self):
        mask = [[-1,    1,    1],
                [-1,   -1,    1],
                [-1,    1,   -1],
                [-1,   -1,   -1],
                [ 1,   -1,   -1],
                [-1,   -1,   -1],
                [-1,    1,    1]]

        gt_classes = [2, 2, 3]
        self.mask = np.asarray(mask)
        self.gt_classes = np.asarray(gt_classes)
        self.class_mask = self.mask * self.gt_classes

    def test_classmask(self):
        answer = [[-2,    2,    3],
                  [-2,   -2,    3],
                  [-2,    2,   -3],
                  [-2,   -2,   -3],
                  [ 2,   -2,   -3],
                  [-2,   -2,   -3],
                  [-2,    2,    3]]

        np.testing.assert_array_almost_equal(self.class_mask, answer)

    def test_background(self):
        """
        Description:
        background 라벨이 펴기 되어 있는 onehot vector 을 생성 합니다.
        """
        n_length = tf.shape(tf.constant(self.mask))[0]
        background = tf.zeros(n_length, dtype=tf.int32)
        background = tf.one_hot(background, 4)

        #         0  1  2  3
        answer = [[1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 0]]
        answer = np.asarray(answer)
        sess = tf.Session()
        background_ = sess.run(background)
        np.testing.assert_array_almost_equal(background_, answer)

    def test_indices(self):
        positive_index = tf.where(self.class_mask > 0)
        indices = positive_index[:, 0]

        answer_indices = np.asarray([0, 0, 1, 2, 4, 6, 6])

        sess = tf.Session()
        indices_ = sess.run(indices)
        np.testing.assert_array_almost_equal(answer_indices, indices_)

    def test_indices(self):
        positive_index = tf.where(self.class_mask > 0)
        positive_value = tf.gather_nd(self.class_mask, positive_index)

        answer = np.asarray([2, 3, 3, 2, 2, 2, 3])

        sess = tf.Session()
        positive_value_ = sess.run(positive_value)
        np.testing.assert_array_almost_equal(answer, positive_value_)

    def test_classmask_onehot(self):
        positive_index = tf.where(self.class_mask > 0)
        positive_value = tf.gather_nd(self.class_mask, positive_index)

        positive_onehot = tf.one_hot(positive_value, 4)
        answer = [[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        answer = np.asarray(answer)

        sess = tf.Session()
        positive_onehot_ = sess.run(positive_onehot)
        np.testing.assert_array_almost_equal(positive_onehot_, answer)

    def test_scatter_nd(self):
        n_length = tf.shape(tf.constant(self.mask))[0]
        background = tf.zeros(n_length, dtype=tf.int32)
        background = tf.one_hot(background, 4)

        positive_index = tf.where(self.class_mask > 0)
        positive_value = tf.gather_nd(self.class_mask, positive_index)
        positive_onehot = tf.one_hot(positive_value, 4)

        indices = positive_index[:, 0]
        indices = tf.expand_dims(indices, axis=-1)
        pred = tf.tensor_scatter_nd_update(background, indices, positive_onehot)

        sess = tf.Session()
        pred_ = sess.run(pred)

        answer = [[0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]]

        np.testing.assert_array_almost_equal(pred_, answer)

    def test_generate_trainble_classes(self):
        pred_classes = generate_trainble_classes(tf.constant(self.mask),
                                                 tf.constant(self.gt_classes), 4)


        sess = tf.Session()
        pred_classes_ = sess.run(pred_classes)

        answer = [[0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]]

        np.testing.assert_array_almost_equal(answer, pred_classes_)
