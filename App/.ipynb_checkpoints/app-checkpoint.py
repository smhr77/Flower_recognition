from flask import Flask, render_template, request
import pickle

model = pickle.load(open("flower.pkl", 'rb'))


app = Flask(__name__)

@app.route('/', methods= ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods =['POST'])
def predict():
    imagefile = request.files['imagefile']


if __name__  == '__main__':
    app.run(port =3000, debug = True)
