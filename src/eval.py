import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import cv2

from model.config import model_path, inferImgPath, inferImgPath, img_size,\
    sampleSubmission, submission_test
from model.unet import unet
from src.utils import mask2rle


def eval_net():
    if not os.path.exists(model_path):
        raise Exception('model.h5 not exist!')

    testfiles=os.listdir(inferImgPath)

    unet_model = unet()
    model = unet_model.load_weights(model_path)

    test_img = []
    for fn in tqdm_notebook(testfiles, disable=True):
        img = cv2.imread(os.path.join(inferImgPath, fn))
        img = cv2.resize(img,(img_size,img_size))
        test_img.append(img)

    predict = model.predict(np.asarray(test_img))
    # i = 1
    pred_rle = []
    for img in predict:      
        img = cv2.resize(img, (1600, 256))
        tmp = np.copy(img)
        tmp[tmp<np.mean(img)] = 0
        tmp[tmp>0] = 1
        pred_rle.append(mask2rle(tmp))

    sub = pd.read_csv(sampleSubmission)
    for fn, rle in zip(testfiles, pred_rle):
        sub['EncodedPixels'][sub['ImageId'] == fn] = rle

    sub.to_csv(submission_test, index=False)


if __name__ == "__main__":
    eval_net()
