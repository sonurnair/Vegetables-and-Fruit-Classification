import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained vegetable detection model
model = load_model('model-019.model')

# Initialize the camera
source = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not source.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define labels and colors for vegetables
labels_dict = {
    0: 'Peas', 1: 'Apple', 2: 'Watermelon', 3: 'Jalepeno', 4: 'Onion', 5: 'Pomegranate',
    6: 'Paprika', 7: 'Soy Beans', 8: 'Chilli Pepper', 9: 'Capsicum', 10: 'Tomato',
    11: 'Orange', 12: 'Mango', 13: 'Cucumber', 14: 'Grapes', 15: 'Lettuce', 16: 'Sweetpotato',
    17: 'Pineapple', 18: 'Garlic', 19: 'Carrot', 20: 'Pear', 21: 'Ginger', 22: 'Cabbage',
    23: 'Beetroot', 24: 'Turnip', 25: 'Lemon', 26: 'Corn', 27: 'Raddish', 28: 'Potato',
    29: 'Spinach', 30: 'Bell Pepper', 31: 'Cauliflower', 32: 'Banana', 33: 'Eggplant',
    34: 'Kiwi', 35: 'Sweetcorn'
}

color_dict = {
    0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 255), 4: (255, 0, 255),
    5: (255, 255, 0), 6: (0, 255, 255), 7: (255, 255, 255), 8: (128, 0, 128), 9: (0, 128, 0),
    10: (255, 165, 0), 11: (255, 69, 0), 12: (0, 255, 0), 13: (135, 206, 250), 14: (255, 20, 147),
    15: (0, 191, 255), 16: (139, 69, 19), 17: (255, 255, 0), 18: (128, 128, 128),
    19: (255, 165, 0), 20: (255, 255, 255), 21: (0, 0, 255), 22: (0, 128, 0),
    23: (255, 0, 0), 24: (255, 105, 180), 25: (255, 223, 0), 26: (255, 140, 0),
    27: (255, 20, 147), 28: (139, 0, 139), 29: (34, 139, 34), 30: (0, 255, 255),
    31: (255, 99, 71), 32: (255, 255, 255), 33: (0, 255, 255), 34: (255, 20, 147),
    35: (255, 69, 0)
}

# Create the face detector (or another object detection model if necessary)
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, img = source.read()

    # Check if the frame is properly captured
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # You can replace this with an actual object detection method, if necessary
    # For simplicity, I'm still using face detection as a placeholder
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        # Crop the region of interest (to simulate object detection)
        veg_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(veg_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))  # Assuming grayscale model input

        # Predict the vegetable class
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding boxes and labels for detected vegetables
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the result
    cv2.imshow('Vegetable Detection', img)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

# Release the camera and close OpenCV windows
source.release()
cv2.destroyAllWindows()
