# https://medium.com/@dustindavignon/upload-multiple-images-with-python-flask-and-flask-dropzone-d5b821829b1d
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import logging
import random
import time

from PIL import Image
import requests, os
from io import BytesIO

from predict import TransferModel

from settings import *

app = Flask(__name__)
dropzone = Dropzone(app)

# print(os.getcwd())
# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/transfer/uploads'
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

# model
resnet_model = TransferModel('transfer/', 'resnet152_final_model.pt')
hrnet_model = TransferModel('transfer/', 'hrnet_final_model.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            print (file.filename)

            t = time.time() # get execution time

            # request.files['photo'].save('test.jpg')
            filename = photos.save(
                file,
                folder='photos',
                name=file.filename    
            )
            # append image urls
            print(f'saving to {photos.url(filename)}')
            file_urls.append(photos.url(filename))

            image = Image.open(file)

            class_name, class_probability = hrnet_model.predict(image)
            hrnet_prediction = f'HRNet predicted {class_name} with a probability of {round(class_probability,4)*100}%'
            print(hrnet_prediction)

            class_name, class_probability = resnet_model.predict(image)
            resnet_prediction = f'ResNet predicted {class_name} with a probability of {round(class_probability,4)*100}%'
            print(resnet_prediction)

            dt = time.time() - t
            app.logger.info("Execution time: %0.02f seconds" % (dt))


        session['file_urls'] = file_urls
        session['hrnet_prediction'] = hrnet_prediction
        session['resnet_prediction'] = resnet_prediction
        return "uploading and predicting..."
    return render_template('index.html')

@app.route('/results')
def results():
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)

    resnet_prediction = session['resnet_prediction']
    session.pop('resnet_prediction', None)

    hrnet_prediction = session['hrnet_prediction']
    session.pop('hrnet_prediction', None)

    return render_template('results.html', file_urls=file_urls, resnet_prediction=resnet_prediction, hrnet_prediction=hrnet_prediction)

@app.route('/predict', methods=['GET'])
def predict():
    
    # response = requests.get(url)
    # img = open_image(BytesIO(response.content))
    img = None

    t = time.time() # get execution time

    class_name, class_probability = resnet_model.predict(img)

    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))

    print(f'predicting {class_name} with a probability of {round(class_probability,4)*100}%')

    return jsonify({"class_name": class_name, "class_probability": class_probability})

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'memcached'
    app.run(host="0.0.0.0", debug=True, port=PORT)
