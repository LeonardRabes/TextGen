import tensorflow as tf
import numpy as np
import time
from create_model import create_model


NAME = "textgen_rnn_{}".format(int(time.time()))
TRAIN_DATA = "train_data.npy"
TRAIN_LABEL = "train_label.npy"
TEST_DATA = "test_data.npy"
TEST_LABEL = "test_label.npy"

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

xtrain = np.load(TRAIN_DATA)
ytrain = np.load(TRAIN_LABEL)
xtest = np.load(TEST_DATA)
ytest = np.load(TEST_LABEL)

print(ytrain[0])
model = create_model()

model.fit(xtrain, ytrain, epochs=100, callbacks=[tensorboard])

val_loss, val_acc = model.evaluate(xtest, ytest)
print(val_loss)
print(val_acc)

model.save(NAME + ".model")
