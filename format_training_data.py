import os
import numpy
import tensorflow as tf

def format_data(filename):
    CHAR_PER_BATCH = 10

    text = open(filename, mode='rb').read()

    data = []
    label = []

    #create batches
    for i in range(len(text)):
        if i + CHAR_PER_BATCH < len(text):
            batch = []
            #to int
            for char in text[i:i+CHAR_PER_BATCH]:
                batch.extend(byte2bin(char))


            data.append(batch)
            label.append(text[i+CHAR_PER_BATCH])
        else:
            print("End of Stream!")
            break

    border = int(len(label) * 0.85)

    numpy.save("train_data.npy", data[:border])
    numpy.save("train_label.npy", label[:border])

    numpy.save("test_data.npy", data[border:])
    numpy.save("test_label.npy", label[border:])


def byte2bin(byte):
    bits = [0, 0, 0, 0, 0, 0, 0, 0]
    for index, bit in enumerate(reversed(bin(byte)[2:])):
        bits[7-index] = int(bit)

    return bits


def bin2byte(bits):
    byte = 0
    for i, bit in enumerate(reversed(bits)):
        if bit:
            byte = byte + 2 ** i
    return byte


format_data("speeches.txt")
