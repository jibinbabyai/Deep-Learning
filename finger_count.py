import mediapipe as mp
import cv2
import pyttsx3 

engine = pyttsx3.init()
voice=engine.getProperty('voices')
engine.setProperty('voice',voice[1].id)
engine.setProperty('volume',0.9)

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils


hands = mphands.Hands(max_num_hands = 1)

video = cv2.VideoCapture(0)

counter = 0
tip = [8,12,16,20]


def gestures(gest):
    all_gest = {
        (0,0,0,0,0): "Fist",
        (1,1,1,1,1): "Hiii",
        (0,1,0,0,0): "Point",
        (1,0,0,0,0): "Thumbs up",
        (0,1,1,0,0): "Peace",
        (1,1,0,0,1): "Rock"
    }
    return all_gest.get(tuple(gest),"Unknown")

while True:
    suc,raw = video.read()
    img = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    lmlist = []
    finger = [5,5,5,5,5]
    

    if results.multi_hand_landmarks:
        hand_label = results.multi_handedness[0].classification[0].label
        for handms in results.multi_hand_landmarks:
            for id,lm in enumerate(handms.landmark):
                
                lmlist.append([id,lm.x,lm.y])
            mpdrawing.draw_landmarks(raw,handms,mphands.HAND_CONNECTIONS,mpdrawing.DrawingSpec(color = (0,0,255),thickness = 1))

    if len(lmlist)>0 and hand_label == 'Right':
        finger = []
        if lmlist[4][1] < lmlist[4-2][1]:
            finger.append(1)
        else:
            finger.append(0)
        for i in tip:
            if lmlist[i][2] < lmlist[i-2][2]:
                finger.append(1)
            else:
                finger.append(0)
    count = finger.count(1)
    print(finger)
    

    gest = gestures(finger)
            
    cv2.putText(raw,f"Gesture :{gest}",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    if counter % 15 == 0 and gest != 'Unknown':
        engine.say(gest)
        engine.runAndWait()
    counter +=1
    
    cv2.imshow("Finger Count",raw)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

