
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from helpers import preprocess
import traceback
import cv2

INPUT_SHAPE = (224,224,3)

loaded_model = load_model('classifier_netb3.h5')

def predict(image, filename):
    try:
        # 1. Resize image
        resized_image = cv2.resize(image, INPUT_SHAPE[:2])

        # save resized image
        file_path = "./static/images/" + filename
        cv2.imwrite(file_path, resized_image)

        # Preprocess image
        features = preprocess(resized_image)

        img_array = keras.preprocessing.image.img_to_array(features['cleaned_image'])
        img_array = tf.expand_dims(img_array, 0)

        # Run inference
        predictions = loaded_model.predict(img_array)

        # Make predictions 
        probability = predictions[0][0]
        predicted_class = 'malignant' if probability > 0.5 else 'benign'

        border_irregularity = 'high' if features['border_irregularity'] < 0.25 else 'low'

        asym_index = (features['horizontal_dissymmetry'] + features['vertical_dissymmetry']) / 2
        shape_asymmetry = 'low' if asym_index > 0.25 else 'high'

        colour_asymmetry = 'high' if features['color_features']['color_asymmetry'] > 80 else 'low'

        score = float(predictions[0])
        ratio = f"This image is {100 * (1 - score):.2f}% benign and {100 * score:.2f}% malignant."

        result = {
            'predictions': {
                'message': ratio,
                'predicted_class': predicted_class,
                'shape_asymmetry': shape_asymmetry,
                'border_irregularity':  border_irregularity,
                'colour_asymmetry': colour_asymmetry
            },
            'contoured_lesion': features['contoured_image']
        }
        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
        return 'Lesion has not been detected on the image. Please try again'
