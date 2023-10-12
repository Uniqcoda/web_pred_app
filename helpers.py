import cv2
import numpy as np
import math

# Constants
INPUT_SHAPE = (224,224,3)
ASYMMETRY_THRESHOLD = 8  # Minimum contour area for asymmetry calculation
STRUCTURING_ELEMENT_SIZE = 6 # For hair detection
MIN_CONTOUR_AREA = 700 # More than this will not be able to detect very small lesions
MAX_CONT_DISTANCE_RATIO = 0.4

# ***** Clean image ***** 
def remove_hair(img):
    """
    Remove hair from image using morphological transformation, Gaussian blur and inpainting.

    :param img: 3-d image (resized to standard)
    :return: Cleaned image with no hair
    """
    try:
        # Convert the image to grayscale
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Create a rectangular kernel filled with ones
        kernel = cv2.getStructuringElement(1, (STRUCTURING_ELEMENT_SIZE, STRUCTURING_ELEMENT_SIZE)) 

        # Black hat filter
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

        # Gaussian filter
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

        # Binary thresholding (replacing grey areas with black or white)
        ret, hair_mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

        # Replace pixels of the mask with original image
        cleaned_image = cv2.inpaint(img, hair_mask, 6, cv2.INPAINT_TELEA)

        return cleaned_image
    except Exception as e:
        print("Function 'remove_hair' Error:", e)
        return None

# ***** Get Masks *****
def get_mask(image):
    """
    Get image mask without holes in them
    :param image: 3-d image
    :return: 2-d mask
    """
    try:
        # Convert the cleaned image to grayscale 
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply thresholding to convert the image to binary format using THRESH_OTSU to choose the best threshold
        # After this operation, all the pixels below mean_pixel value will be 0, and all the pixels above mean_pixel will be 255
        ret, disease_mask = cv2.threshold(img_gray, 0 , 255, cv2.THRESH_OTSU)

        # Invert the grayscale mask so the disease is covered in white
        disease_mask = cv2.bitwise_not(disease_mask)

        # Fill the holes in the mask
            # Get the disconnected regions by finding contours
        filled_mask = np.copy(disease_mask)
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Fill disconnected regions with white
        for cnt in contours:
            cv2.drawContours(filled_mask, [cnt], 0, 255, -1)

        return filled_mask
    except Exception as e:
        print("Function 'get_mask' Error:", e)
        return None


# ***** Get contoured lesion *****
def segment_lesion(mask2d, image, min_contour_area=MIN_CONTOUR_AREA):
    """
    Extract the largest contour of the image, to ensure that the lesion is covered in the contour,
    as there may be multiple dark shades on the image.

    :param mask2d: 2-d mask of the gray image.
    :param image: 3-d image on which the contour lines would be drawn.
    :param min_contour_area: Minimum area to consider a contour as valid (to filter out small contours).
    :return: the image on which the contour was drawn, the largest contour (list of contour points of the lesion),
             and the mask of the lesion.
    """
    try:
        image_copy = np.copy(image)
        contours, _ = cv2.findContours(mask2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter out small contours that might be noise or not relevant
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

        if not valid_contours:
            # If no valid contours are found, return the original image with no contour drawn
            return image_copy, None, None

        # Sort the valid contours by area in descending order
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        # Calculate the center of the image
        image_center_x, image_center_y = image_copy.shape[1] // 2, image_copy.shape[0] // 2

        # Calculate the max contour to center distance allowed 
        max_dist =  MAX_CONT_DISTANCE_RATIO * min(image_copy.shape[0], image_copy.shape[1]) 


        # Filter out contours that are too close to the image edges,
            # as they may be just image vignette
        filtered_contours = []
        for contour in sorted_contours:
            M = cv2.moments(contour)
            contour_center_x = int(M["m10"] / M["m00"])
            contour_center_y = int(M["m01"] / M["m00"])

            # Calculate the distance from the contour center to the image center
            distance_to_center = np.sqrt((contour_center_x - image_center_x) ** 2 + (contour_center_y - image_center_y) ** 2)

            # Check if the distance is within the allowed range
            if distance_to_center <= max_dist:
                filtered_contours.append(contour)

        if not filtered_contours:
            # If no valid contours are found after filtering,
            print('No lesion detected close to the center of the image') 
            return image_copy, None, None
        else:
            largest_contour = filtered_contours[0]

        # Draw the largest contour on the image using red blue lines
        cv2.drawContours(image_copy, [largest_contour], -1, (255, 0, 0), 2) 

        # Create the mask based on the largest contour
        largest_mask = np.zeros_like(mask2d)
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return image_copy, largest_contour, largest_mask
    except Exception as e:
        print("Function 'segment_lesion' Error:", e)
        return image_copy, None, None


# ***** Get lesion features ABCD *****
# EXTRACT DIMENSIONS
def extract_dimensions(mask, largest_contour):
    """
    Extract Diameter feature, along with lesion area, centroid, perimeter,
    and transformed image using affine transformation.

    :param mask: 2D binary image mask.
    :param largest_contour: list of contour points of the lesion.
    :return: a dictionary containing all the features along with area, centroid, perimeter
             of the lesion, asymmetry, border irregularity, and transformed image.
    """
    try:
        contour_area = cv2.countNonZero(mask)
    
        # moments = cv2.moments(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)

        # Calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = 2 * radius

        features = {
            'area': int(contour_area),
            'perimeter': int(contour_perimeter),
            'diameter': diameter,
        }

        return features

    except Exception as e:
        print("Function 'extract_dimensions' Error:", e)
        return {}


# ASYMMETRY
def get_asymmetry_features(image, mask, largest_contour, contour_area):
    """
    Calculate the degree of asymmetry (by getting the dissymmetry) of the lesion based on the warped mask. 
    (explained on the last paragraph page 2, ABDER-RAHMAN et al, 2020)

    :param image: 3D image of the lesion.
    :param mask: 2D binary image mask.
    :param largest_contour: largest_contour of the lesion.
    :param contour_area: area of the lesion.
    :return: Degree of dissymmetry for both horizontal and vertical axes.
    """
    ROTATION_SCALE = 1.0

    try:
        # Rotate the lesion for horizontal and vertical analyses
        ellipse = cv2.fitEllipse(largest_contour)
        (ellipse_x, ellipse_y), (ellipse_w, ellipse_h), ellipse_angle = ellipse

        if ellipse_w < ellipse_h:
            if ellipse_angle < 90:
                ellipse_angle -= 90
            else:
                ellipse_angle += 90

        rows, cols = mask.shape

        rotation_matrix = cv2.getRotationMatrix2D((ellipse_x, ellipse_y), ellipse_angle, ROTATION_SCALE)
        warped_mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))
        warped_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        warped_image_segmented = cv2.bitwise_and(warped_image, warped_image, mask=warped_mask)

        contours, _ = cv2.findContours(warped_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        contour = contours[np.argmax(areas)]
        x, y, nW, nH = cv2.boundingRect(contour)
        warped_mask = warped_mask[y:y + nH, x:x + nW]

        # Flip the mask horizontally (like flipping the pages of a book)
        flipped_mask_horizontal = cv2.flip(warped_mask, 1)
        # Flip the mask vertically
        flipped_mask_vertical = cv2.flip(warped_mask, 0)

        # Compare the flipped masks to the original mask, and get the difference (as  black pixels)
        diff_horizontal = cv2.compare(warped_mask, flipped_mask_horizontal, cv2.CV_8UC1)
        diff_vertical = cv2.compare(warped_mask, flipped_mask_vertical, cv2.CV_8UC1)

        # Convert the black pixels to white 
        diff_horizontal = cv2.bitwise_not(diff_horizontal)
        diff_vertical = cv2.bitwise_not(diff_vertical)

        # Count the number of non-zero areas of the diff mask (white pixels)
        horizontal_dissymmetry = cv2.countNonZero(diff_horizontal)
        vertical_dissymmetry = cv2.countNonZero(diff_vertical)

        # Normalize dissymmetry features by the area (to get an index)
        normalized_horizontal_dissymmetry = round(float(horizontal_dissymmetry) / contour_area, 2)
        normalized_vertical_dissymmetry = round(float(vertical_dissymmetry) / contour_area, 2)

        return {
            'horizontal_dissymmetry': normalized_horizontal_dissymmetry,
            'vertical_dissymmetry': normalized_vertical_dissymmetry,
            'warped_mask': warped_mask,
            'warped_image_segmented': warped_image_segmented
        }
    except Exception as e:
        print("Function 'get_asymmetry_features' Error:", e)
        return None
    
# BORDER IRREGULARITY
def get_border_irregularity(perimeter, area):
    """
    Calculate the border irregularity of the lesion using the inverse of compactness index I=P**2/4πA.
    
    
    :param perimeter: the contour perimeter of the lesion mask.
    :param area: the contour area of the lesion mask.
    :return: Border irregularity
    """
    pie = math.pi
    c_index = round(perimeter**2 / (4 * area * pie), 2)
    return 1/c_index

# COLOUR
def get_color_features(lesion_region):
    """
    Get color-related features from the lesion image.

    :param image: 3D numpy array of an RGB image.
    :param mask: Binary image of the lesion mask.
    :return: Dictionary of color-related features.
    """
    try:
        # Convert the lesion region to LAB color space
        lab_lesion = cv2.cvtColor(lesion_region, cv2.COLOR_RGB2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_lesion)

        # Calculate the standard deviation of each channel
        l_std = np.std(l_channel)
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)

        # Calculate the mean and standard deviation of the L, A, and B channels
        mean_l = np.mean(l_channel)
        mean_a = np.mean(a_channel)
        mean_b = np.mean(b_channel)

        # Calculate the color variance using the standard deviations
        color_variance = np.sqrt(l_std**2 + a_std**2 + b_std**2)

        # Calculate the color intensity as the mean of the standard deviations
        color_intensity = (l_std + a_std + b_std) / 3.0

        # Calculate the color asymmetry as the absolute difference between the mean
        # of the L channel and the mean of the A and B channels
        color_asymmetry = np.abs(mean_l - (mean_a + mean_b) / 2.0)

        return {
            'color_variance': round(color_variance, 2),
            'color_intensity': round(color_intensity, 2),
            'color_asymmetry': round(color_asymmetry, 2)
        }
    except Exception as e:
        print("Function 'get_color_features' Error:", e)
        return {}

def preprocess(image):
    # 1. Resize image
    resized_image = cv2.resize(image, INPUT_SHAPE[:2])

    # 2. Clean image
    cleaned_image = remove_hair(resized_image)

    # 3. Get contoured lesion
    masks = get_mask(cleaned_image)
    contoured_image, largest_contour, single_mask = segment_lesion(masks, cleaned_image)

    # 4. Get lesion features ABCD
    features = extract_dimensions(single_mask, largest_contour)
    lesion_region = cv2.bitwise_and(cleaned_image, cleaned_image, mask=single_mask)

    # Calculate Calculate ABCD features if the contour area is greater than the threshold
    if  features['area'] > ASYMMETRY_THRESHOLD:
        # Calculate asymmetry (A)
        asymmetry_features = get_asymmetry_features(cleaned_image, single_mask, largest_contour, features['area'])

        # Calculate border feature (B)
        border_irregularity = get_border_irregularity(features['perimeter'], features['area'])

        # Get color feature (C)
        color_features = get_color_features(lesion_region)
    else:
        raise ValueError("No contour detected. Cannot calculate features.") 

    # Add more features to the image features dict
    features['original_image'] = image
    features['cleaned_image'] = cleaned_image
    features['mask2d'] = single_mask
    features['segmented_lesion'] = lesion_region
    features['contoured_image'] = contoured_image 
    features['horizontal_dissymmetry'] = asymmetry_features['horizontal_dissymmetry']
    features['vertical_dissymmetry'] = asymmetry_features['vertical_dissymmetry']
    features['warped_mask'] = asymmetry_features['warped_mask']
    features['warped_image_segmented'] = asymmetry_features['warped_image_segmented']
    features['border_irregularity'] = border_irregularity
    features['color_features'] = color_features
    features['largest_contour'] = largest_contour

    return features
