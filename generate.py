import tensorflow as tf
import numpy as np
from format_training_data import bin2byte, byte2bin

def generate(model, init, length):
    offset = 80 - (len(init) * 8)
    sample = []
    string = init

    for i in range(offset):
        sample.append(0)
    for char in init:
        sample.extend(byte2bin(ord(char)))

    for i in range(length):
        prediction = model.predict(np.array([sample]))
        result = np.argmax(prediction[0])
        string = string + chr(result)
        sample = sample[8:] + byte2bin(result)

    return string


model = tf.keras.models.load_model("textgen_rnn_1560888955.model")
out = generate(model, "I am a ", 800)
print(out)
