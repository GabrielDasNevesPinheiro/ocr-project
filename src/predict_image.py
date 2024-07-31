from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from os import path
import numpy as np
import cv2

img_height, img_width = 28, 28
dataset_path = path.abspath(r".\dataset")
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image

def display_image(image, segments, title):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in segments:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(output)
    plt.title(title)
    plt.show()

def segment_lines_chars(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [cv2.boundingRect(c) for c in contours]
    lines = sorted(lines, key = lambda x: x[1])

    display_image(image, lines, "Imagem Segmentada")

    segments = []
    for (x, y, w, h) in lines:
        line_img = image[y:y + h, x:x + w]
        char_contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chars = [cv2.boundingRect(c) for c in char_contours]
        chars = sorted(chars, key = lambda x: x[0])

        line_segment = [line_img[char_y:char_y + char_h, char_x:char_x + char_w] for (char_x, char_y, char_w, char_h) in chars]
        segments.append(line_segment)
    
    return segments



model = load_model(path.join(path.abspath("./"), "model", "output.h5"))

def predict_character(image_segment, model):
    img = cv2.resize(image_segment, (img_height, img_width))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    return class_labels[class_idx]

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    lines_segments = segment_lines_chars(preprocessed_image)
    
    text = ""
    for line in lines_segments:
        for char_img in line:
            char = predict_character(char_img, model)
            text += char
        text += "\n"
    
    return text.strip()

image_path = path.join(path.abspath("./"), "test_image.png")
extracted_text = extract_text_from_image(image_path)
print(extracted_text)