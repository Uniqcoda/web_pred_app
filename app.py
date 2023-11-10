from flask import Flask, render_template, request 
import traceback
from prediction import predict
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        # Get uploaded image
        file = request.files['image']

        # Read image as OpenCV image
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        result = predict(img, file.filename)

        lesion_path = './static/images/lesion_' + file.filename
        cv2.imwrite(lesion_path, result['contoured_lesion'])

        return render_template('predict.html', prediction = result['predictions'], image_path = lesion_path)
         
    except Exception as e:
        print(e)
        traceback.print_exc()

        return render_template('predict.html', error = 'Lesion has not been detected on the image. Please try again')



if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=False, host='0.0.0.0')

