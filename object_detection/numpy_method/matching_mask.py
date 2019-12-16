import numpy as np

def matching_mask(iou_matrix, iou_threshold=0.7):
    """
    Description:
        조건들에 합치 하는 anchor 에 대한 mask 을 생성합니다.
        조건에 합치 하는 anchor 는 1로
        조건에 합치 하지 않는  anchor 는 -1로 표시

        조건 1. iou 가 threshold 이상인 anchor
        조건 2. 각 ground truth 당 가장 iou 가 큰 anchor

    Args:
        iou_matrix: Ndarray, 2D array

            [[anchor1_gt1 , anchor1_gt2,  anthor1_gt3],
             [anchor2_gt1  , anchor2_gt2,  anthor2_gt3],
             [anchor3_gt1  , anchor3_gt2,  anthor3_gt3],
             [anchor4_gt1  , anchor4_gt2,  anthor4_gt3],
                            ...
            [anchor5_gt1  , anchor5_gt2,  anthor5_gt3]]

        iou_threshold: int, 해당 threshold 이상이면 1 을 표기
        해당 threshold 미만이면 -1 을 표기



    Return:
        matching_mask : Ndarray, 2D array

        [[1 ,   1,  -1], <=anchor1
         [-1 , -1,  -1], <=anchor2
         [-1 , -1,  -1], <=anchor3
               ...
         [1 ,   1,   1]] <=anchor5
         gt1   gt2   gt3

    """
    # threshold_mask, 2D array
    # 1 positive , -1 negative
    mask = np.where(iou_matrix >= iou_threshold, 1, -1)

    # best_indices, 1D array
    best_indices = np.argmax(iou_matrix, axis=0)

    # [5,6] -> [[5, 6], [0, 1]]
    indices_2d = np.stack([best_indices,
                           np.arange(len(best_indices))])

    # MUST change Ndarray to *list*
    indices_2d = indices_2d.tolist()

    mask[indices_2d] = 1
    return mask
