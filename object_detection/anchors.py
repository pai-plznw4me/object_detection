import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def generate_anchor(input_tensor,
                    backbone_output,
                    anchor_default_sizes=(32., 64., 128.),
                    anchor_ratio=(0.5, 1, 2)):
    """
    Description:
        Anchors 을 생성합니다

    Args:
        :param input_tensor: Keras Layer  , 4D Tensor
        :param backbone_output: Keras Layer , 4D Tensor
        :param anchor_default_sizes
        :param anchor_ratio

        :return: anchor_grid: Tensor, 3D Tensor
    """
    # input shape
    input_h = K.shape(input_tensor)[1]
    input_w = K.shape(input_tensor)[2]

    # backbone shape
    backbone_h = K.shape(backbone_output)[1]
    backbone_w = K.shape(backbone_output)[2]

    # to calculate the distance btw feature map pixels
    stride_h = 2. ** tf.ceil(tf.log(input_h / backbone_h)/tf.log(2.))
    stride_w = 2. ** tf.ceil(tf.log(input_w / backbone_w)/tf.log(2.))

    # generate anchor sizes
    n_anchor_sizes = len(anchor_default_sizes) * len(anchor_ratio)
    anchor_sizes = []
    for size in anchor_default_sizes:
        for r in anchor_ratio:
            anchor_sizes.append([size*np.sqrt(r), size/np.sqrt(r)])
    anchor_sizes = np.asarray(anchor_sizes)

    # generate anchor grid
    # 4 => cx, cy, w, h
    fmap_grid = tf.ones(shape=[backbone_h, backbone_w], dtype=tf.float64)

    # generate coordinate center_x, center_y
    range_h = tf.range(backbone_h)
    range_w = tf.range(backbone_w)
    cx, cy = tf.meshgrid(range_w, range_h)
    cx = tf.cast(cx, tf.float64)
    cy = tf.cast(cy, tf.float64)

    # shift cx ,cy
    # pixel_gap//2 은 stride 때문에 저렇게 된다.
    # pixel 간 거리는 stride 만큼 떨어져 있다.
    cx = cx * stride_w + stride_w // 2
    cy = cy * stride_h + stride_h // 2

    # cx 는 anchor 갯수만큼 있어서 저렇게 만든다
    grid_cx = tf.stack([cx] * n_anchor_sizes, axis=-1)
    grid_cy = tf.stack([cy] * n_anchor_sizes, axis=-1)

    # mapping ws, hs to anchor grid
    anchor_ws = anchor_sizes[:, 0]
    anchor_hs = anchor_sizes[:, 1]
    grid_ws = tf.expand_dims(fmap_grid, axis=-1) * anchor_ws
    grid_hs = tf.expand_dims(fmap_grid, axis=-1) * anchor_hs

    """
    Description:
        grid_cx shape = (7,7,9), 
        grid_cx[0, 0, :] => [x1,x2,x3 .. ] 

        grid_cy = shape = (7,7,9)                 [[x1, x2, x3, ...]
        grid_cy[0, 0, :] => [y1,y2,y3 .. ]         [y1, y2, y3, ...]
                                            ==>    [w1, w2, w3, ...]
        grid_ws = shape = (7,7,9)                  [h1, h2, h3, ...]]
        grid_ws[0, 0, :] => [w1,w2,w3 .. ] 

        grid_hs = shape = (7,7,9)
        grid_hs[0, 0, :] => [h1,h2,h3 .. ] 
    """
    anchor_grid = tf.stack([grid_cx, grid_cy, grid_ws, grid_hs], axis=-1)

    """
    Description:
    [[x1, x2, x3, ...]
     [y1, y2, y3, ...]
     [w1, w2, w3, ...]  => [x1,y1,w1,h1, x2,y2,w2,h2 ...] 
     [h1, h2, h3, ...]]

    """
    anchor_grid = tf.reshape(anchor_grid, [backbone_h, backbone_w, -1])
    return anchor_grid


def generate_trainable_anchors(normalize_anchors, matching_mask):
    """
    Args:
        normalize_anchors: 3D array, shape = [N_anchor, N_gt, 4]

        matching_mask: Ndarray, 2D array,
            anchor 로 사용할 것은 *1*로
            anchor 로 사용하지 않을 것은 *-1* 로 표기
            example: [[ 1 ,-1], <-anchor1
                      [-1 ,-1], <-anchor2
                      [-1 ,-1], <-anchor3
                      [ 1 , 1], <-anchor4
                      [-1 , 1]] <-anchor5
                       gt1 gt2
           위 예제에서 사용할 anchor 는 (gt1, anchor1), (gt2, anchor4), (gt2, anchor5)

    Description:

        학습시킬수 있는 anchors을 생성합니다.
        입력된 normalize_anchors 는 Shape 을 [N_anchor, N_gt, 4] 가집니다.
        위 normalize_anchors 에서 학습해야 할 anchor 을 추출합니다.
        최종 return 될 anchor 는 [N_acnhor , 4] 의 shape 을 가집니다.


        해당 vector 에서 postive_mask 에 표시된(1로 표기된) 좌표의
        anchor 만 가져옵니다.
        해당 anchor 을 가져와 shape 가 [N_anchor , 4] 인 anchor 에 넣습니다.

        # Caution! #
        만약 가져올 anchor 가 없으면 (예제 anchor3) -1 -1 -1 -1로 채운다
        만약 가져올 anchor 가 많다면 가장 오른쪽에 있는 (gt2, anchor4) anchor 을 선택한다.


    """
    # Tensorflow
    # TODO 여기서 mathcing_mask == 1 을 하면 Error 가 발생된다. 그 이유는?
    indices_2d = tf.where(tf.equal(matching_mask, 1))
    indices_2d = tf.stack(indices_2d, axis=0)

    indices = indices_2d[:, 0]
    indices = tf.expand_dims(indices, axis=-1)

    # calculate delta
    # [0] 을 붙이는 이유는 tf.gather_nd 을 사용하고 나면 출력 tensor의 shape 가 (1, N, 4) 로 나온다
    # 1 은 필요없어 제거하기 위해 [0]을 붙인다
    dx = tf.gather_nd(normalize_anchors[:, :, 0], [indices_2d])[0]
    dy = tf.gather_nd(normalize_anchors[:, :, 1], [indices_2d])[0]
    dw = tf.gather_nd(normalize_anchors[:, :, 2], [indices_2d])[0]
    dh = tf.gather_nd(normalize_anchors[:, :, 3], [indices_2d])[0]

    d_xywh = tf.stack([dx, dy, dw, dh], axis=-1)

    n_anchors = tf.shape(normalize_anchors)[0]
    ret_anchor = tf.ones([n_anchors, 4], dtype=tf.float32) * -1
    ret_anchor = tf.tensor_scatter_nd_update(ret_anchor, indices, d_xywh)
    return ret_anchor


def generate_trainble_classes(mask, gt_classes, n_classes):
    """
    Description:

    Args:
        mask: Tensor, 2D array
            example:
                [[1 , -1,  1],
                 [1 , -1,  1],
                 [1 , -1,  1],
                 [1 , -1,  1],
                     ...
                 [1 , -1,  1]]

        gt_classes: Tensor, 1D vector
            example:
                [2, 2, 3]


    Return:
        matching_mask : Tensor, 2D array
        example:
            [[ 2,   2,  -3],
             [-2 , -2,  -3],
             [-2 , -2,  -3],
                   ...
             [2 ,   2,   3]]
    """
    class_mask = mask * gt_classes

    n_length = tf.shape(class_mask)[0]
    background = tf.zeros(n_length, dtype=tf.int32)
    background = tf.one_hot(background, n_classes)

    positive_index = tf.where(class_mask > 0)
    positive_value = tf.gather_nd(class_mask, positive_index)
    positive_onehot = tf.one_hot(positive_value, n_classes)

    indices = positive_index[:, 0]
    indices = tf.expand_dims(indices, axis=-1)

    pred_classes = tf.tensor_scatter_nd_update(background, indices, positive_onehot)


    return pred_classes
