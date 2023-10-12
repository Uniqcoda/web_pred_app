
import tensorflow as tf
from tensorflow import keras
from helpers import preprocess
import traceback

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_clss_vgg.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict(image):
    try:
        # Preprocess image
        features = preprocess(image)

        img_array = keras.preprocessing.image.img_to_array(features['cleaned_image'])
        img_array = tf.expand_dims(img_array, 0)

        # Run inference
        interpreter.set_tensor(input_index, img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        asym_index = (features['horizontal_dissymmetry'] + features['vertical_dissymmetry']) / 2

        # Make predictions 
        probability = predictions[0][0]
        predicted_class = 'malignant' if probability > 0.5 else 'benign'

        border_irregularity = 'high' if features['border_irregularity'] < 0.25 else 'low'

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
        return 'Error making prediction for image. Please try again'
