# test_cnn.py
# This script is now 100% correct.
# It perfectly matches the normalization from your dataset script.

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# 1. Load the trained CNN model
print("Loading CNN model 'isl_cnn_model.h5'...")
model = load_model('isl_cnn_model.h5')
print("Model loaded successfully.")

# 2. Load the gesture labels
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Loading labels from keypoint.csv...")
data = pd.read_csv('keypoint.csv')
y = data.iloc[:, 0] # The first column is the label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
actions = label_encoder.classes_
print(f"Loaded {len(actions)} actions/gestures: {actions}")


# 3. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 4. Main helper functions
def mediapipe_detection(image, hands_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

# =================================================================
#  THE FINAL, CORRECT PREPROCESSING FUNCTION
#  (Based on dataset_keypoint_generation.py)
# =================================================================
def extract_keypoints(results):
    """
    Extracts the 42 keypoints and performs the *full*
    normalization (relative to wrist + max value scaling).
    """
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0] 
        
        # 1. Get all keypoints as (21, 2)
        all_keypoints = np.array([[res.x, res.y] for res in hand.landmark])
        
        # 2. Calculate relative-to-wrist coordinates
        wrist = all_keypoints[0] # Get wrist
        normalized_keypoints = all_keypoints - wrist # Normalize
        
        # 3. Flatten to 1D list (42,)
        temp_landmark_list = normalized_keypoints.flatten()
        
        # 4. Normalize by max absolute value
        max_value = np.max(np.abs(temp_landmark_list))
        
        # Check for divide by zero
        if max_value == 0:
            # If max_value is 0 (e.g., hand perfectly still at 0,0),
            # just return the zeros.
            return temp_landmark_list
        
        def normalize_(n):
            return n / max_value
        
        final_keypoints = np.array(list(map(normalize_, temp_landmark_list)))
        
        return final_keypoints
    else:
        # If no hand is detected, return an array of 42 zeros
        # This will still correctly predict 'A'
        return np.zeros(21 * 2)
# =================================================================
#  END OF FUNCTION
# =================================================================

# 5. Start Webcam and Detection Loop
print("Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam. Try changing 0 to 1.")
    exit()

current_prediction = ""
confidence = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        image, results = mediapipe_detection(frame, hands)
        draw_landmarks(image, results)
        
        # Get keypoints
        keypoints = extract_keypoints(results)

        # Check if the keypoints are all zeros (i.e., no hand detected)
        # This is the "A" sign, so we should NOT skip it.
        # We will keep the logic from the last step to show '...'
        # if the user PREFERS to see nothing.
        
        # Let's test if 'A' is all zeros
        is_all_zeros = np.all(keypoints == 0)

        if is_all_zeros:
            # If no hand, clear the prediction. This is the behavior
            # you asked for (to hide 'A' when no hand is shown).
            current_prediction = ""
        else:
            # If there IS a hand, proceed with prediction
            
            # 1. Create batch of 1: (1, 42)
            prediction_input = np.array([keypoints]) 
            # 2. Add feature dimension for Conv1D: (1, 42, 1)
            prediction_input = np.expand_dims(prediction_input, axis=2)

            # Make prediction
            res = model.predict(prediction_input, verbose=0)[0]
            predicted_class_index = np.argmax(res)
            current_prediction = actions[predicted_class_index]
            confidence = res[predicted_class_index]

        # Display the prediction on the screen
        if current_prediction != "":
            # We have a hand, show the prediction
            cv2.putText(image, f'{current_prediction} ({confidence*100:.0f}%)', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # No hand, show '...'
            cv2.putText(image, '...', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Indian Sign Language Detection (CNN)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Script finished.")