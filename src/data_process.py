import os
import random
import pandas as pd
import matplotlib as plt
import numpy as np
import cv2

from model.config import trainImgPath, img_size, inferImgPath
from src.utils import mask2rle, rle2mask


class DataLoader:
    
    def __init__(self, csv):
        self.pd = pd.read_csv(csv)
        self.df_train = self.pd[self.pd['EncodedPixels'].notnull()].reset_index(drop=True)

    def overview(self):
        return self.df_train.head()

    def class_analise(self):
        class_dict = self.df_train["ClassId"].value_counts()
        
        palet = [(250, 230, 20), (30, 200, 241), (200, 30, 250), (250,60,20)]

        fig, ax = plt.subplots(1, 4, figsize=(6, 2))
        for i in range(1, 5):
            ax[i].axis('off')
            ax[i].imshow(np.ones((10, 40, 3), dtype=np.uint8) * palet[i])
            ax[i].set_title(f"class:{i}, count:{class_dict[i]}")

        plt.show()
        
    def mask_compre_dispaly(self):
        r = random.randint(1, 100)
        fn = self.df_train['ImageId'].iloc[r]
        img = cv2.imread(os.path.join(trainImgPath, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
        
        mask = rle2mask(self.df_train['EncodedPixels'].iloc[r], img.shape)
        
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        x_batch += [img]
        y_batch += [mask]
        
        plt.imshow(x_batch[0])
        plt.imshow(np.squeeze(y_batch[0]))

    def infer_display(self, pred):
        r = random.randint(1, 100)

        testfiles=os.listdir(inferImgPath)
        img_t = cv2.imread(os.path.join(inferImgPath, testfiles[r]))
        plt.imshow(img_t)
        
        img = pred[r]
        img = cv2.resize(img, (1600, 256))
        tmp = np.copy(img)
        tmp[tmp<np.mean(img)] = 0
        tmp[tmp>0] = 1
        _pred_img = mask2rle(tmp)
        plt.imshow(_pred_img)