import urllib.request
import numpy as np
import cv2
from PIL import Image


def sample_dataset():
    sample_name = '342.jpg'

    image_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/pascal/images/{}".format(sample_name)
    urllib.request.urlretrieve(image_url, "sample_image.jpg")

    mask_url = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/pascal/roidb/{}.npy".format(sample_name)
    urllib.request.urlretrieve(mask_url, "sample_label.npy")

    image_path = "sample_image.jpg"
    label_path = "sample_label.npy"

    # load sample image
    sample_img = np.asarray(Image.open(image_path).convert('RGB'))
    sample_img=sample_img/255.
    sample_tensor = np.expand_dims(sample_img, axis=0)

    # load label
    sample_bboxes = np.load(label_path).astype(np.int32)

    # draw bounding boxes
    for bbox in sample_bboxes:
        print(bbox)
        patched_image = sample_tensor[0].copy()
        cv2.rectangle(patched_image,
                                      (bbox[0], bbox[1]),
                                      (bbox[2], bbox[3]),
                                      (255,0,0),
                                      10)
    return sample_tensor, sample_bboxes