import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

# --- Configuration Constants ---

# Smoothing factor (alpha in LPF) - balances responsiveness (low value) vs. stability (high value) [10]
SMOOTHENING = 7

# Default frame size for capturing video
FRAME_W, FRAME_H = 640, 480

# Get screen resolution dynamically [9]
SCREEN_W, SCREEN_H = pyautogui.size()

# Variables for the Low-Pass Filter (LPF) [10]
# P_LOC: Previous cursor location
# C_LOC: Current cursor location
P_LOC_X, P_LOC_Y = 0, 0
C_LOC_X, C_LOC_Y = 0, 0

# Margin to define the active area in the webcam feed (to avoid cursor instability at edges) [11]
AREA_REDUCTION = 100

# Distance (in pixels) between index and thumb tips to trigger a click [7]
CLICK_DISTANCE_THRESHOLD = 30

# Landmark IDs for all finger tips (for identifying extended fingers)
TIP_IDS = [4, 8, 12, 16, 20] # 4:Thumb, 8:Index, 12:Middle, 16:Ring, 20:Pinky
INDEX_TIP = 8
THUMB_TIP = 4

# --- Initialization ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# Initialize MediaPipe Hands solution [12]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_up_fingers(lmList):
    """
    Checks which fingers are extended (up) based on landmark positions.
    Returns a list: (1=up, 0=down) [9]
    """
    fingers = []
    
    # Thumb check (specific logic due to thumb's lateral movement)
    # Checks if thumb tip (4) is further left/right than the base joint (2)
    if lmList[THUMB_TIP][1] > lmList[THUMB_TIP - 1][1]:
        fingers.append(1) # Thumb up
    else:
        fingers.append(0) # Thumb down

    # 4 Fingers check (Index, Middle, Ring, Pinky)
    # Checks if the tip landmark's Y-coordinate is higher (smaller value) than the joint below it [13]
    for id in range(1, 5):
        # lmList][7] is the Y-coordinate of the tip
        # lmList[7] is the Y-coordinate of the joint below the tip
        if lmList[TIP_IDS[id]][2] < lmList[TIP_IDS[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers 

# --- Main Program Loop ---
while True:
    success, img = cap.read()
    if not success:
        break

    # 1. Image Pre-processing
    img = cv2.flip(img, 1) # Flip for mirror effect [7]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Hand Landmark Detection
    results = hands.process(img_rgb)
    lmList = [] # Stores for 21 landmarks

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw the hand skeleton
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract all 21 landmarks
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates [14]
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

    if lmList:
        # 3. Get Hand and Finger Information
        x1, y1 = lmList[8][1], lmList[8][2] # Index finger tip coordinates (x, y)
        x2, y2 = lmList[4][1], lmList[4][2] # Thumb tip coordinates (x, y)

        fingers = get_up_fingers(lmList) #

        # Define the active working area for visual feedback [11]
        cv2.rectangle(img, (AREA_REDUCTION, AREA_REDUCTION),
                      (FRAME_W - AREA_REDUCTION, FRAME_H - AREA_REDUCTION),
                      (255, 0, 255), 2)

        # 4. Mouse Control Logic

        # Mode 1: Moving Mode (Index Finger Up ONLY)
        # Check if Index is up (fingers[1]==1) and all others are down (sum of Middle, Ring, Pinky is 0)
        # The thumb check is less strict here for ease of movement
        if fingers[1] == 1 and sum(fingers[2:]) == 0:

            # 4.1. Map Hand Position to Screen Position
            # Interpolate the clamped hand position (x1, y1) onto the screen resolution [14]
            clamped_x = np.interp(x1, (AREA_REDUCTION, FRAME_W - AREA_REDUCTION), (0, SCREEN_W))
            clamped_y = np.interp(y1, (AREA_REDUCTION, FRAME_H - AREA_REDUCTION), (0, SCREEN_H))

            C_LOC_X = clamped_x
            C_LOC_Y = clamped_y

            # 4.2. Apply Smoothing (Low-Pass Filter) [10]
            # final_x/y is the smoothed position
            final_x = P_LOC_X + (C_LOC_X - P_LOC_X) / SMOOTHENING
            final_y = P_LOC_Y + (C_LOC_Y - P_LOC_Y) / SMOOTHENING

            # 4.3. Move Mouse [9]
            pyautogui.moveTo(final_x, final_y)

            P_LOC_X, P_LOC_Y = final_x, final_y # Update previous location for the next iteration

            # Visual feedback: Highlight index tip (indicating move mode)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # Mode 2: Left Click Mode (Index and Thumb Pinch)
        # Calculate distance between Index Tip (8) and Thumb Tip (4) [9]
        distance = math.hypot(x1 - x2, y1 - y2)

        # If Index is up (needed for activation context) AND the pinch distance is close
        if fingers[1] == 1 and distance < CLICK_DISTANCE_THRESHOLD:
            # Highlight pinch zone (indicating click mode)
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)

            # Perform Left Click (pyautogui handles the click action) [9]
            # Note: PyAutoGUI is fast enough that repeated clicks are handled correctly.
            pyautogui.click()
            print("Left Click Activated")

    # 5. Display the frame
    cv2.imshow("AI Virtual Mouse", img)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()