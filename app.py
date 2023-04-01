from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import random
import time
from keras.applications.imagenet_utils import  decode_predictions
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_io as tfio

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH = 'models/ad-48m.h5'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print('Model loaded. Check http://127.0.0.1:5000/')

def get_unique_name():
    timestamp = int(time.time())
    rand_num = random.randint(1, 1000)
    return f"{timestamp}_{rand_num}"

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def show_spectrogram(spectrogram):
    plt.figure(figsize=(30, 20))
    plt.imshow(tf.transpose(spectrogram)[0])
    plt.show()

def model_predict(file_path, model):
    spectrogram = preprocess(file_path)
    # display spectrogram for the result.
    spectrogram = np.expand_dims(spectrogram, axis=0)
    prediction = model.predict(spectrogram)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(tf.transpose(spectrogram)[0])
    ax.axis("off")
    file_name = "spec_"+get_unique_name()
    fig.savefig("uploads/" + file_name, bbox_inches="tight")

    return (1 if prediction > 0.8 else 0, file_name)

from flask import send_file


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file('uploads/' + filename, mimetype='image/png')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['audio']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds, spectrogram  = model_predict(file_path, model)
        print(spectrogram)
        return render_template("index.html", preds=preds, spectrogram=spectrogram)
    return None  



if __name__ == "__main__":
    app.run(debug=True)