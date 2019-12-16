import numpy as np

def generate_trainable_anchors(normalize_anchors, matching_mask):
    """
    Args:
        normalize_anchors: 3D array, shape = [N_anchor, N_gt, 4]

        matching_mask: Ndarray, 2D array,
            anchor 로 사용할 것은 *1*로
            anchor 로 사용하지 않을 것은 *-1* 로 표기
            example: [[1 ,-1], <-anchor1
                      [-1,-1], <-anchor2
                      [-1,-1], <-anchor3
                      [ 1, 1], <-anchor4
                      [-1, 1]] <-anchor5
                       gt1 gt2
           위 예제에서 사용할 anchor 는 (gt1, anchor1), (gt2, anchor4), (gt2, anchor5)

    Description:

        학습시킬수 있는 anchors을 생성합니다.
        입력된 normalize_anchors 는 Shape 을 [N_anchor, N_gt, 4] 가집니다.
        위 normalize_anchors 에서 학습해야 할 anchor 을 추출합니다.
        학습될 anchor 는 [N_acnhor , 4] 의 shape 을 가집니다.


        해당 vector 에서 postive_mask 에 표시된(1로 표기된) 좌표의
        anchor 만 가져옵니다.
        해당 anchor 을 가져와 shape 가 [N_anchor , 4] 인 anchor 에 넣습니다.

        # Caution! #
        만약 가져올 anchor 가 없으면 (예제 anchor3) -1 -1 -1 -1로 채운다
        만약 가져올 anchor 가 많다면 가장 오른쪽에 있는 (gt2, anchor4) anchor 을 선택한다.


    """
    #
    indices_2d = np.where(matching_mask == 1)
    indices_2d = np.stack(indices_2d, axis=0).tolist()

    # trainable axis=0 기준으로 어디에다가 추출한 x,y,w,h 을 넣어야 할지 가리키는 indices
    indices = indices_2d[0]

    # delta 에서 해당 좌표를 가져온다
    dx = normalize_anchors[:, :, 0][indices_2d]
    dy = normalize_anchors[:, :, 1][indices_2d]
    dw = normalize_anchors[:, :, 2][indices_2d]
    dh = normalize_anchors[:, :, 3][indices_2d]

    # stack 한다.
    d_xywh = np.stack([dx, dy, dw, dh], axis=-1)
    #
    ret_anchor = np.ones([len(normalize_anchors), 4], dtype=np.float32) * -1
    ret_anchor[indices] = d_xywh
    return ret_anchor
