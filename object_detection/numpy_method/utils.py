import tensorflow as tf


def anchor_cxcywh2xyxy(anchor_grid):
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

    anchor_grid[:, :, :, 0] = x1_grid
    anchor_grid[:, :, :, 1] = y1_grid
    anchor_grid[:, :, :, 2] = x2_grid
    anchor_grid[:, :, :, 3] = y2_grid

    return anchor_grid


def convert_xyxy2d_to_cxcywh2d(coordinates):
    """
    Description:
        [[x1, y1, x`1, y`1]       [[cx1, cy1, w1, h1]
         [x2, y2, x`2, y`2] =>     [cx2, cy2, w2, h2]
                ...                        ...
         [x_N, y_N, x`_N, y`_N]]   [cx_N, cy_N, w_N, h_N]]

    Args:
        coordinates: Ndarray, 2D array

    Return:
        anchor_grid: Ndarray, 2D array, => NHWC
    """
    w = coordinates[:, 0] - coordinates[:, 2]
    h = coordinates[:, 1] - coordinates[:, 3]
    cx = coordinates[:, 0] + w / 2
    cy = coordinates[:, 0] + h / 2

    coordinates[:, 0] = cx
    coordinates[:, 1] = cy
    coordinates[:, 2] = w
    coordinates[:, 3] = h

    return coordinates