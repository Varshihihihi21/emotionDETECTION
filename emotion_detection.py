import cv2
import numpy as np
from keras.models import model_from_json

def load_model():
    # Load model architecture
    json_file = open("emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    
    # Load model weights
    model.load_weights("emotiondetector.h5")
    return model

# Function to preprocess the image
def extract_features(image):
    # Resize to 48x48
    image_resized = cv2.resize(image, (48, 48))
    
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_resized
    
    # Repeat grayscale channel to make it 3-channel
    image_3channel = cv2.merge([image_gray, image_gray, image_gray])
    
    # Normalize
    image_normalized = image_3channel / 255.0
    
    # Reshape to match model's expected input
    # The model expects (1, 48, 48, 3)
    image_processed = np.expand_dims(image_normalized, axis=0)
    
    return image_processed

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
          4: 'neutral', 5: 'sad', 6: 'surprise'}

def main():
    # Load the model
    model = load_model()

    # Load Haar Cascade for face detection
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    # Open webcam
    webcam = cv2.VideoCapture(0)

    # Check if webcam is opened successfully
    if not webcam.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        # Read frame from webcam
        ret, frame = webcam.read()
        
        # Check if frame is read correctly
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Prepare image for model prediction
            processed_face = extract_features(face_roi)

            # Predict emotion
            pred = model.predict(processed_face)
            emotion = labels[np.argmax(pred)]

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put text of predicted emotion
            cv2.putText(frame, emotion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Emotion Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

# Print model summary to verify input shape
def print_model_summary(model):
    model.summary()

if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    # Print model summary to verify input shape
    print_model_summary(model)
    
    # Run the main detection
    main()