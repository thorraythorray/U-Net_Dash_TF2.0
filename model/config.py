import os


img_size = 256
batch_size = 16
epoch = 20
steps_per_epoch = 100
WORK_PATH = "/Users/xxxx/Datasets/severstal-steel-defect-detection"

trainImgPath = os.path.join(WORK_PATH, "train_images")
trainCsv = os.path.join(WORK_PATH, "train.csv")
inferImgPath = os.path.join(WORK_PATH, "test_images/")
sampleSubmission = os.path.join(WORK_PATH, "sample_submission.csv")
model_path = os.path.join(WORK_PATH, "model.h5")
DefectDetection_history = os.path.join(WORK_PATH, "DefectDetection_history.csv")
submission_test = os.path.join(WORK_PATH, "submission_test.csv")
loss_history_fig = os.path.join(WORK_PATH, "loss_history_fig.jpg")
