
def normalize_anchors(flat_cxcywh_anchors, flat_cxcywh_gt):
    """
    Args:
        flat_cxcywh_gt : Ndarray, 1D arrray, example) [cx, cy, w, h,
                                                     cx1, cy1, w1, h1 ...]

        flat_cxcywh_anchors : Ndarray, 1D arrray, example) [gt_cx, gt_cy, gt_w, gt_h,
                                                gt_cx1, gt_cy1, gt_w1, gt_h1
                                                ...]
    Return:
        norm_anchors: Ndarray, 3D array shape [N_anchor, N_gt, 4]

    Description:
        anchors 와 ground truths 을 통해 거리 계산을 한다.
    """

    # 1d array (N*4) to 2d array (N, 4)
    anchors_2d = np.reshape(flat_cxcywh_anchors, (-1, 4))
    gt_2d = np.reshape(flat_cxcywh_gt, (-1, 4))

    # (N_anchor, 2) -> (1, N_anchor, 2)
    expand_anchors = np.expand_dims(anchors_2d, axis=1)

    # (N_gt, 2) -> (N_gt, 1, 2)
    expand_gt = np.expand_dims(gt_2d, axis=0)

    # Calculate delta
    delta_x = (expand_gt[:, :, 0] - expand_anchors[:, :, 0]) / expand_anchors[:, :, 2]
    delta_y = (expand_gt[:, :, 1] - expand_anchors[:, :, 1]) / expand_anchors[:, :, 3]
    delta_w = np.log(expand_gt[:, :, 2]) - np.log(expand_anchors[:, :, 2])
    delta_h = np.log(expand_gt[:, :, 3]) - np.log(expand_anchors[:, :, 3])

    # Caution #
    # dtype=np.float32 이 구문을 제거하면 default 가 int 이되서 float 값을 대입하면 모두 0이 된다. 주의하자.
    norm_anchors = np.ones_like(expand_anchors - expand_gt, dtype=np.float32)
    norm_anchors[:, :, 0] = delta_x
    norm_anchors[:, :, 1] = delta_y
    norm_anchors[:, :, 2] = delta_w
    norm_anchors[:, :, 3] = delta_h

    return norm_anchors
