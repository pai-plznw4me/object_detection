import tensorflow as tf


def convert_ccwh4d_to_xyxy4d(anchor_grid):
    """
    Args:
        anchor_grid: Ndarray, 4D array, N, H, W, CH
        CH 순서 cx , cy , w, h

    Return:
        anchor_grid: Ndarray, 4D array, => NHWC

    """

    # x1
    x1_grid = anchor_grid[:, :, :, 0] - anchor_grid[:, :, :, 2] / 2

    # x2
    x2_grid = anchor_grid[:, :, :, 0] + anchor_grid[:, :, :, 2] / 2

    # y1
    y1_grid = anchor_grid[:, :, :, 1] - anchor_grid[:, :, :, 3] / 2

    # y2
    y2_grid = anchor_grid[:, :, :, 1] + anchor_grid[:, :, :, 3] / 2

    anchor_grid = tf.stack([x1_grid, y1_grid, x2_grid, y2_grid], axis=-1)

    return anchor_grid


def convert_xyxy2d_to_ccwh2d(coordinates):
    """
    Description:
        [[x1, y1, x`1, y`1]       [[cx1, cy1, w1, h1]
         [x2, y2, x`2, y`2] =>     [cx2, cy2, w2, h2]
                ...                        ...
         [x_N, y_N, x`_N, y`_N]]   [cx_N, y_N, w_N, h_N]]

    Args:
        coordinates: Ndarray, 2D array

    Return:
        anchor_grid: Ndarray, 2D array, => NHWC
    """
    w = coordinates[:, 2] - coordinates[:, 0]
    h = coordinates[:, 3] - coordinates[:, 1]
    cx = coordinates[:, 0] + w / 2
    cy = coordinates[:, 1] + h / 2

    return tf.stack([cx, cy, w, h], axis=-1)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
