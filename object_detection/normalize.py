import tensorflow as tf


def normalize_anchors(flat_cxcywh_anchors, flat_cxcywh_gt):
    """
    Args:
        flat_cxcywh_gt : Tensor, 1D arrray, example) [cx, cy, w, h,
                                                     cx1, cy1, w1, h1 ...]

        flat_cxcywh_anchors : Tensor, 1D arrray, example) [gt_cx, gt_cy, gt_w, gt_h,
                                                gt_cx1, gt_cy1, gt_w1, gt_h1
                                                ...]
    Return:
        norm_anchors: Tensor, 3D array shape [N_anchor, N_gt, 4]

    Description:
        anchors 와 ground truths 을 통해 거리 계산을 한다.
    """

    # 1d array (N*4) to 2d array (N, 4)
    anchors_2d = tf.reshape(flat_cxcywh_anchors, (-1, 4))
    gt_2d = tf.reshape(flat_cxcywh_gt, (-1, 4))

    # (N_anchor, 2) -> (1, N_anchor, 2)
    expand_anchors = tf.expand_dims(anchors_2d, axis=1)

    # (N_gt, 2) -> (N_gt, 1, 2)
    expand_gt = tf.expand_dims(gt_2d, axis=0)

    # Calculate delta
    delta_x = (expand_gt[:, :, 0] - expand_anchors[:, :, 0]) / expand_anchors[:, :, 2]
    delta_y = (expand_gt[:, :, 1] - expand_anchors[:, :, 1]) / expand_anchors[:, :, 3]
    delta_w = tf.log(expand_gt[:, :, 2]) - tf.log(expand_anchors[:, :, 2])
    delta_h = tf.log(expand_gt[:, :, 3]) - tf.log(expand_anchors[:, :, 3])

    # Caution #
    # dtype=np.float32 이 구문을 제거하면 default 가 int 이되서 float 값을 대입하면 모두 0이 된다. 주의하자.
    norm_anchors = tf.stack([delta_x, delta_y, delta_w, delta_h], -1)

    return norm_anchors


def denormalize(normalize_predictions, anchors):
    """
    :param normalize_anchors: Tensor, 4D array
    :param anchors: Tenwor, 1D array
    :return:
    """
    pred_x = normalize_predictions[:, :, :, 0]
    pred_y = normalize_predictions[:, :, :, 1]
    pred_w = normalize_predictions[:, :, :, 2]
    pred_h = normalize_predictions[:, :, :, 3]
    anchor_x = anchors[:, :, :, 0]
    anchor_y = anchors[:, :, :, 1]
    anchor_w = anchors[:, :, :, 2]
    anchor_h = anchors[:, :, :, 3]

    denorm_x = pred_x * anchor_w + anchor_x
    denorm_y = pred_y * anchor_h + anchor_y
    denorm_w = tf.exp(pred_w) * anchor_w
    denorm_h = tf.exp(pred_h) * anchor_h
    denormalize_anchors = tf.stack([denorm_x, denorm_y, denorm_w, denorm_h], -1)

    return denormalize_anchors
