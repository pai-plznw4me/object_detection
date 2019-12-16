import numpy as np

def calculate_iou(flat_xyxy_bboxes, flat_xyxy_gt):
    """
    sample_bboxes : Ndarray, 1D array [x1, x2, y1, y2, x1, x2, y1, y2, ... ]
    sample_bboxes : Ndarray, 1D array [x1, x2, y1, y2, x1, x2, y1, y2, ... ]
    """

    # 1D array to 2D array
    # [x1, x2, y1, y2, x1, x2, y1, y2 ]
    # >>>
    # [[x1, x2, y1, y2],
    # [x1, x2, y1, y2]]

    res_sample_bboxes = flat_xyxy_bboxes.reshape([-1, 4])
    gt_sample_bboxes = flat_xyxy_gt.reshape([-1, 4])

    # Get Area
    area_sample = (res_sample_bboxes[:, 0] - res_sample_bboxes[:, 2]) * (res_sample_bboxes[:, 1]
                                                                         - res_sample_bboxes[:, 3])
    area_gt = (gt_sample_bboxes[:, 0] - gt_sample_bboxes[:, 2]) * (gt_sample_bboxes[:, 1] - gt_sample_bboxes[:, 3])

    # expand dims for using broadcasting
    # (N, 4) -> (N, 1, 4)
    expand_sample = np.expand_dims(res_sample_bboxes, axis=1)
    # (N, 4) -> (1, N, 4)
    expand_gt = np.expand_dims(gt_sample_bboxes, axis=0)

    # search Maximun
    x1y1 = np.where(expand_sample[:, :, :2] > expand_gt[:, :, :2], expand_sample[:, :, :2], expand_gt[:, :, :2])
    # search Minimun
    x2y2 = np.where(expand_sample[:, :, 2:] < expand_gt[:, :, 2:], expand_sample[:, :, 2:], expand_gt[:, :, 2:])

    # get overlay area
    overlay_area = np.prod(x1y1 - x2y2, axis=-1)

    # expand dimension for broadcasting
    expand_area_sample = np.expand_dims(area_sample, axis=-1)

    iou = overlay_area / (expand_area_sample + area_gt - overlay_area)

    return iou
