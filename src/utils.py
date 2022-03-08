import numpy as np
import os
import cv2

from model.config import trainImgPath


def rle2mask(rle, imgshape):
    # rle编码转化mask编码，展示分割效果
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start): int(start + lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def mask2rle(img):
    tmp = np.rot90(np.flipud(img), k=3)
    rle = []
    lastColor = 0;
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1, 1)   
    for i in range(len(tmp)):
        if (lastColor == 0) and tmp[i] > 0:
            startpos = i
            lastColor = 1
        elif (lastColor == 1) and (tmp[i] == 0):
            endpos = i - 1
            lastColor = 0
            rle.append(str(startpos) + ' ' + str(endpos - startpos + 1))
    return " ".join(rle)


def keras_generator(df_train, batch_size, img_size):
    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            fn = df_train['ImageId'].iloc[i]
            img = cv2.imread(os.path.join(trainImgPath, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            
            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)
            
            img = cv2.resize(img, (img_size, img_size))
            mask = cv2.resize(mask, (img_size, img_size))

            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)
