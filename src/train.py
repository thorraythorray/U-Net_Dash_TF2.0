import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint

from model.unet import unet
from src.utils import keras_generator
from src.data_process import DataLoader
from model.config import trainCsv, batch_size, img_size, model_path, \
    DefectDetection_history, loss_history_fig, epoch, steps_per_epoch


def train_net():
    pd_data = DataLoader(trainCsv)
    df_train = pd_data.df_train
    train_df, val_df = train_test_split(df_train, test_size=0.15)
    train_generator = keras_generator(train_df, batch_size, img_size)
    validator_generator = keras_generator(val_df, batch_size, img_size)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model = unet()
    results = model.fit_generator(train_generator, 
                                steps_per_epoch=steps_per_epoch,
                                epochs=epoch, 
                                validation_data=validator_generator,
                                validation_steps = len(val_df)/batch_size,
                                verbose=2,
                                shuffle=True,
                                callbacks=callbacks_list)

    # Save history (for next Resume)
    hist_df = pd.DataFrame(results.history)[['loss','val_loss']]
    hist_df.to_csv(DefectDetection_history, index=False)

    # Plot
    # fig, ax = plt.subplots(1,1,figsize=(15, 8))
    # ax.plot(hist_df['loss'], color='b', label="Training loss")
    # ax.plot(hist_df['val_loss'], color='r', label="validation loss",axes=ax)
    # ax.legend(loc='best', shadow=True)
    # plt.savefig(loss_history_fig)
    
    return results

if __name__ == "__main__":
    train_net()
