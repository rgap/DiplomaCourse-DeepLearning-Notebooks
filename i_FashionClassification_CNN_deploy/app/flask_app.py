
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as image_utils

RUTA = '/home/rgap/mysite/'

app = Flask(__name__)


def saveImage(img):
    img.save(RUTA + img.filename)
    return img.filename

def preprocesamiento(img):
    imgPre = image_utils.img_to_array(img)
    imgPre = imgPre/255
    imgPre = 1 - imgPre
    return imgPre
@app.route('/predict/', methods = ['POST'])
def predict():
    json_file = open(RUTA + 'modelo.json', 'r')
    modelo_json = json_file.read()
    json_file.close()

    modeloConv = model_from_json(modelo_json)
    modeloConv.load_weights(RUTA + 'modelo.h5')
    modeloConv.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    imagen = saveImage(request.files['imagen'])
    x = image_utils.load_img(RUTA + imagen, target_size = (28,28), color_mode = 'grayscale')
    x2 = preprocesamiento(x)

    prediccion = modeloConv.predict(x2.reshape(1,28,28,1))
    categoria = np.argmax(prediccion)

    return render_template('index.html', clase = categoria)

@app.route('/')
def index():
    return render_template('index.html')

