from flask import Flask, render_template, request 
import traceback
from prediction import predict
import cv2

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        # Get uploaded image
        file = request.files['image']
        file_path = "./static/images/" + file.filename
        file.save(file_path)
        # Read the image           
        img = cv2.imread(file_path)

        result = predict(img)

        lesion_path = './static/images/lesion_' + file.filename
        cv2.imwrite(lesion_path, result['contoured_lesion'])

        return render_template('predict.html', prediction = result['predictions'], image_path = lesion_path)
         
    except Exception as e:
        print(e)
        traceback.print_exc()

        return render_template('predict.html', error = 'Error making prediction for image. Please try again')



if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host='0.0.0.0')

