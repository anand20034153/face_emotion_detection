import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
from keras.models import model_from_json, Sequential  # Import Sequential from keras.models
from keras import layers
from PIL import Image, ImageTk

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('C:/Users\ANAND S\Documents\project completed\p-work-emotion-detection\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Register the necessary Keras layers to avoid missing class errors
from keras import layers

# Load the model structure
emotion_model = model_from_json(loaded_model_json, custom_objects={"Sequential": Sequential, 
                                                                "Conv2D": layers.Conv2D, 
                                                                "MaxPooling2D": layers.MaxPooling2D, 
                                                                "Dropout": layers.Dropout, 
                                                                "Flatten": layers.Flatten, 
                                                                "Dense": layers.Dense, 
                                                                "InputLayer": layers.InputLayer})

# Load weights into the model
emotion_model.load_weights('C:/Users\ANAND S\Documents\project completed\p-work-emotion-detection\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main/model/emotion_model.h5')
print("Loaded model from disk")

# Set up the Tkinter window
root = tk.Tk()
root.title("Emotion Detection")

# Set up the Canvas widget to display video frames and make it responsive to window size
canvas = Canvas(root)
canvas.pack(fill="both", expand=True)

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('C:/Users\ANAND S\Documents\project completed\p-work-emotion-detection\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main/haarcascades/haarcascade_frontalface_default.xml')

# Check if Haar Cascade is loaded correctly
if face_detector.empty():
    print("Error: Haar Cascade file not loaded correctly. Check the path.")
    exit()

def update_frame():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        return
    
    # Get the current size of the Canvas (window size)
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Resize the frame to fit the current canvas size
    frame = cv2.resize(frame, (canvas_width, canvas_height))

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face detected
    for (x, y, w, h) in num_faces:
        # Draw rectangle around the face (this is just for visualization)
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Extract the region of interest (ROI) for emotion prediction
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        
        # Preprocess the face image (resize and expand dimensions)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Convert the frame to RGB for displaying in Tkinter (Tkinter uses RGB, OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL Image object
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img)

    # Update the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Keep a reference to the image object to prevent garbage collection
    canvas.image = img_tk

    # Call the update_frame function every 10 milliseconds (this is to update the frame every 100 FPS)
    root.after(10, update_frame)

# Call the update_frame function to start displaying the frames
update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
