import cv2
import mediapipe as mp
import time

# Function to check if thumb up gesture is detected
def is_thumb_up_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    thumb_up = thumb_tip.y < index_tip.y
    return thumb_up

# Initialize VideoCapture.
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands with custom confidence thresholds.
mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # Adjust the confidence thresholds here.

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #num of hands
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            mpDraw.draw_landmarks(img, handLMS, mp_hands.HAND_CONNECTIONS)
            
            # Check for thumb up gesture
            thumb_up = is_thumb_up_gesture(handLMS)
            if thumb_up:
                cv2.putText(img, "Good job!", (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release the VideoCapture and destroy all windows
cap.release()
cv2.destroyAllWindows()
