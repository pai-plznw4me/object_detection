import tensorflow as tf


def calculate_iou(flat_xyxy_bboxes, flat_xyxy_gt):
    """
    Args:
        flat_xyxy_bboxes : Tensor, 1D array [x1, y1, x2, y2, x1, y1, x2, y2, ... ]
        flat_xyxy_gt : Tensor, 1D array [gt_x1, gt_y1, gt_x2, gt_y2, gt_x1, gt_y1, gt_x2, gt_y2, ... ]
    Return:
        iou: Tensor, 2D array
        array shape (N_anchor, N_gt)
    """

    # 1D array to 2D array
    # [x1, x2, y1, y2, x1, x2, y1, y2 ]
    # >>>
    # [[x1, x2, y1, y2],
    # [x1, x2, y1, y2]]

    res_sample_bboxes = tf.reshape(flat_xyxy_bboxes, shape=([-1, 4]))
    gt_sample_bboxes = tf.reshape(flat_xyxy_gt, shape=([-1, 4]))

    # Get Area
    area_sample = (res_sample_bboxes[:, 0] - res_sample_bboxes[:, 2]) * (res_sample_bboxes[:, 1]
                                                                         - res_sample_bboxes[:, 3])
    area_gt = (gt_sample_bboxes[:, 0] - gt_sample_bboxes[:, 2]) * (gt_sample_bboxes[:, 1] - gt_sample_bboxes[:, 3])

    # expand dims for using broadcasting
    # (N_anchor, 4) -> (N_anchor, 1, 4)
    expand_sample = tf.expand_dims(res_sample_bboxes, axis=1)

    # (N_gt, 4) -> (1, N_gt, 4)
    expand_gt = tf.expand_dims(gt_sample_bboxes, axis=0)

    # Tensorflow 에서는 broadcasting 기능이 where 에서 제공되지 않는다.
    # 그래서 이렇게 우회하는 방법을 쓴다.
    broadcast_frame = expand_sample[:, :, :] - expand_gt[:, :, :]
    broadcast_sample = tf.ones_like(broadcast_frame) * expand_sample[:, :, :]
    broadcast_gt = tf.ones_like(broadcast_frame) * expand_gt[:, :, :]

    # search Maximun
    x1y1 = tf.where(expand_sample[:, :, :2] > expand_gt[:, :, :2],
                    broadcast_sample[:, :, :2],
                    broadcast_gt[:, :, :2])

    # search Minimun
    x2y2 = tf.where(expand_sample[:, :, 2:] < expand_gt[:, :, 2:],
                    broadcast_sample[:, :, 2:],
                    broadcast_gt[:, :, 2:])

    # get overlay area
    diff_area = x2y2 - x1y1 + 1
    diff_area = tf.where(diff_area < 0, tf.zeros_like(diff_area), diff_area)
    overlay_area = tf.reduce_prod(diff_area, axis=-1)

    # expand dimension for broadcasting
    # gt 가 여러개 있을수 있어서 이렇게 broadcasting  을 해줘 한번에 연산한다
    expand_area_sample = tf.expand_dims(area_sample, axis=-1)

    iou = overlay_area / (expand_area_sample + area_gt - overlay_area)

    return iou
