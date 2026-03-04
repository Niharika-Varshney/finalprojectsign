import mediapipe as mp
import cv2 as cv
import os
import pickle

# Initializing MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize lists to store data and labels
data = []
labels = []

# Directory containing the dataset
datadir = 'Data_BSL'

print("[INFO] Starting dataset processing...")

# Loop through each directory in the dataset directory
for dir_ in os.listdir(datadir):
    print(f"\n[INFO] Processing folder: {dir_}")
    img_count = 0
    for img_path in os.listdir(os.path.join(datadir, dir_)):
        aux = []
        img = cv.imread(os.path.join(datadir, dir_, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    aux.append(x)
                    aux.append(y)
            data.append(aux)
            labels.append(dir_)
        img_count += 1
        if img_count % 50 == 0:
            print(f"[INFO] {img_count} images processed in folder '{dir_}'...")

print("\n[INFO] Dataset processing complete!")
print(f"[INFO] Total samples collected: {len(data)}")

# Save the data and labels
with open('data_BSL.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("[INFO] Data saved to 'data_BSL.pickle'")
