from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('flower.pkl', 'rb'))
app = Flask(__name__)


@app.route('/', methods= ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods =['POST'])
def predict():
    imagefile= request.files['imagefile']
    from keras.preprocessing import image
    test_image = image.load_img(imagefile, model)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    return render_template('index.html')


if __name__  == '__main__':
    app.run(port =3000, debug = True)
