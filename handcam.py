import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up video capture
cap = cv2.VideoCapture(0)


# Function to detect if hand is closed
def is_hand_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.05  # Adjust threshold as needed


# Main loop
try:
    while True:
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw hand landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand is closed
                if is_hand_closed(hand_landmarks):
                    pyautogui.press('space')
                    time.sleep(0.5)  # Delay to avoid multiple presses

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
