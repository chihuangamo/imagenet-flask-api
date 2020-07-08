from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import expand_dims
from keras.applications.inception_v3 import preprocess_input
import json
from PIL import Image
import urllib.request
import numpy as np


app = Flask(__name__)
api = Api(app)

model = load_model('inceptionv3.h5') 
## Model loaded from keras
# from keras.applications import InceptionV3
# model = InceptionV3(
#     include_top=True,
#     weights="imagenet",
#     classes=1000
# )
with open('imagenet_class_index.json', 'r') as f:
    labels = json.load(f)

def get_image(image_url):
    with urllib.request.urlopen(image_url) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())

def predict():
    image_size = (299, 299)
    img = load_img('temp.jpg', target_size=image_size)
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)  
    img_preprocessed = preprocess_input(img_array)

    predictions = model.predict(img_preprocessed, steps=1)
    return predictions

def get_best_prediction(predictions, n):
    top = predictions[0].argsort()[::-1][:n]
    res = {}
    for i in top:
        res[labels[str(i)][1]] = '{:.3f}'.format(predictions[0][i])
    return res


class ImageClassify(Resource):
    def post(self):
        data = request.get_json()
        url = data['url']
        n = data['n']
        
        try:
            get_image(url)
            predictions = predict()
            res = get_best_prediction(predictions, n)
            print(res)
            return jsonify({
                'status': 200,
                'classification': res
            })
        except:
            return jsonify({
                'status': 303,
                'message': 'Something went wrong. You may try another image'
            })

api.add_resource(ImageClassify, '/classify')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)