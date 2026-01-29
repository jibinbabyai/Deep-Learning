import mediapipe as mp
import cv2
import math
import subprocess
import math
import time
import random
import numpy as np
import threading



speech_thread = None
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh()
pose = mp_pose.Pose()

video = cv2.VideoCapture(0)






PIPER_BIN = "/home/devjith/Luminar/Deep Learning/piper/piper/piper"
VOICE_MODEL = "/home/devjith/Luminar/Deep Learning/piper/voices/en_US-hfc_female-medium.onnx"
OUT_WAV = "/tmp/piper_test.wav"

p_marks = [0,11,12,23,24]





INTERVIEW_QUESTIONS = {

    "introduction": [
        "Please introduce yourself.",
        "Could you briefly introduce yourself?",
        "Tell me a little about yourself.",
        "Can you give a short introduction about yourself?",
        "Let’s start with a brief self-introduction."
    ],

    "behavioral": [
        "Tell me about a challenging project you worked on and how you handled it.",
        "Describe a time when you had to learn a new technology quickly.",
        "How do you handle tight deadlines or pressure?",
        "Have you ever faced a failure in a project? What did you learn from it?",
        "How do you handle feedback or criticism?"
    ],

    "machine_learning_basics": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain bias–variance tradeoff.",
        "What is overfitting and how can you prevent it?",
        "What evaluation metrics do you use for classification problems?",
        "Explain the concept of gradient descent."
    ],

    "deep_learning": [
        "What is the vanishing gradient problem?",
        "Explain the difference between CNNs and RNNs.",
        "What are activation functions and why are they needed?",
        "What is batch normalization?",
        "How does dropout help in neural networks?"
    ],

    "nlp": [
        "What is tokenization and why is it important?",
        "Explain the difference between stemming and lemmatization.",
        "What are word embeddings?",
        "How do transformers work at a high level?",
        "What is attention mechanism?"
    ],

    "computer_vision": [
        "What is the difference between image classification and object detection?",
        "Explain how convolution works in CNNs.",
        "What is OpenCV used for?",
        "What are keypoints and feature descriptors?",
        "How does MediaPipe detect pose landmarks?"
    ],

    "projects": [
        "Explain one of your major projects in detail.",
        "What problem were you trying to solve in you project?",
        "What challenges did you face during implementation?",
        "How did you evaluate your model’s performance?",
        "If you had more time, what improvements would you make?"
    ],

    "system_design_ai": [
        "How would you design a real-time posture detection system?",
        "How would you handle latency in a computer vision pipeline?",
        "How do you deploy a machine learning model in production?",
        "What are the challenges in real-time AI applications?",
        "How would you scale an AI system?"
    ],

    "hr_closing": [
        "Why should we hire you?",
        "What are your strengths and weaknesses?",
        "Where do you see yourself in five years?",
        "Are you open to learning new technologies?",
        "Do you have any questions for us?"
    ]
}

question_categories = [
    "introduction",
    "behavioral",
    "machine_learning_basics",
    "deep_learning",
    "nlp",
    "computer_vision",
    "projects",
    "system_design_ai",
    "hr_closing"
]



def speak_async(text):
    global speech_thread

    if speech_thread and speech_thread.is_alive():
        return  

    speech_thread = threading.Thread(
        target=speak,
        args=(text,),
        daemon=True
    )
    speech_thread.start()

def speak(text):
    process = subprocess.Popen(
        [PIPER_BIN, "--model", VOICE_MODEL, "--length_scale","1.05","--output_file", OUT_WAV],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    process.stdin.write(text.encode("utf-8"))
    process.stdin.close()
    process.wait()

    subprocess.run(["aplay", OUT_WAV])


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])



def angle_between(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = max(min(dot / (mag1 * mag2), 1), -1)
    return math.degrees(math.acos(cos_angle))




def midpoint(p1, p2):
    return (
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
        (p1.z + p2.z) / 2
    )

def calculation(lm):
    nose = lm[0]
    ls, rs = lm[11], lm[12]
    lh, rh = lm[23], lm[24]

    sh_center = midpoint(ls, rs)
    hip_center = midpoint(lh, rh)

    shoulder_width = abs(ls.x - rs.x)
    if shoulder_width < 1e-4:
        return None

   
    shoulder_vec = (rs.x - ls.x, rs.y - ls.y)
    shoulder_slope_angle = abs(angle_between(shoulder_vec, (1, 0)))

   
    torso_vec = (
        sh_center[0] - hip_center[0],
        sh_center[1] - hip_center[1]
    )
    chest_angle = angle_between(torso_vec, (0, -1))

  
    spine_vec = (
        hip_center[0] - sh_center[0],
        hip_center[1] - sh_center[1]
    )
    spine_angle = angle_between(spine_vec, (0, 1))

    nose_x_norm = (nose.x - sh_center[0]) / shoulder_width

    return shoulder_slope_angle, chest_angle, spine_angle, nose_x_norm



    
base_shoulder = []
base_chest = []
base_spine = []
base_nose = []

sh_deviation = []
ch_deviation = []
sp_deviation = []
nose_deviation = []

sh_filter_deviation = []
ch_filter_deviation = []
sp_filter_deviation = []
nose_filter_deviation = []




counter = 0





def pose_capture(question):
    global counter
    global base_shoulder, base_chest, base_spine, base_nose
    global sh_deviation, ch_deviation, sp_deviation, nose_deviation
    global sh_filter_deviation, ch_filter_deviation, sp_filter_deviation, nose_filter_deviation

    time_now = time.time()

    while True:
        time_diff = time.time() - time_now
        if time_diff >= 60:
            return True
            
        suc,raw = video.read()

        if not suc:
            print("Cant read camera!! Try Again")
            speak_async("Cant read camera!! Try Again")

            break
        img = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        pose_detected = pose.process(img)


        if pose_detected.pose_landmarks:
            
            mp_drawing.draw_landmarks(raw,pose_detected.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            lm  = pose_detected.pose_landmarks.landmark

            if counter < 25:
                res = calculation(lm)
                if res is None:
                    continue
                b_sh, b_ch, b_sp, b_nose= res
                base_shoulder.append(b_sh)
                base_chest.append(b_ch)
                base_spine.append(b_sp)
                base_nose.append(b_nose)
                counter += 1
                continue




            sh_base = sum(base_shoulder) / len(base_shoulder)
            chest_base = sum(base_chest) / len(base_chest)
            spine_base = sum(base_spine) / len(base_spine)
            nose_base = sum(base_nose) / len(base_nose)


            pose_measure = calculation(lm)
            if pose_measure is None:
                continue
            sh_pose, ch_pose, sp_pose, nose_pose = pose_measure

            sh_deviation.append(sh_pose - sh_base)
            ch_deviation.append(ch_pose - chest_base)
            sp_deviation.append(sp_pose - spine_base)
            nose_deviation.append(nose_pose - nose_base)

            if len(sh_deviation) >= 30:

                sh_filter_deviation.append(sum(sh_deviation) / len(sh_deviation))
                ch_filter_deviation.append(sum(ch_deviation) / len(ch_deviation))
                sp_filter_deviation.append(sum(sp_deviation) / len(sp_deviation))
                nose_filter_deviation.append(sum(nose_deviation) / len(nose_deviation))


                sh_deviation,ch_deviation,sp_deviation,nose_deviation = [],[],[],[]
                
        
        cv2.putText(raw,question,(25,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),thickness=2)
        cv2.putText(raw,f"Time Left : {int(60 - time_diff)}",(25,95),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),thickness=2)
        cv2.imshow('Capture Frame',raw)
       
        cv2.waitKey(1)
    cv2.destroyAllWindows()
        
                

q_count = 0
for i in question_categories:
    
    question = INTERVIEW_QUESTIONS[i][random.randint(0,4)]
    print(question)
    speak_async(question)
    res = pose_capture(question)
    q_count+=1
    if q_count>=9:
        speak("Thank You for attending..... Please wait for results!")
        

video.release()
cv2.destroyAllWindows()


    






def first25mean(dev):
    if len(dev) == 0:
        return 0
    first25 = dev[:int(0.25 * len(dev))]
    return sum(abs(x) for x in first25) / len(first25)

def last25mean(dev):
    if len(dev) == 0:
        return 0
    last25 = dev[-(int(0.25 * len(dev))):]
    return sum(abs(x) for x in last25) / len(last25)

def safe_avg(arr):
    return sum(abs(x) for x in arr) / len(arr) if arr else 0.0


sh_first_25_mean = first25mean(sh_filter_deviation)
ch_first_25_mean = first25mean(ch_filter_deviation)
sp_first_25_mean = first25mean(sp_filter_deviation)
np_first_25_mean = first25mean(nose_filter_deviation)

sh_last_25_mean = last25mean(sh_filter_deviation)
ch_last_25_mean = last25mean(ch_filter_deviation)
sp_last_25_mean = last25mean(sp_filter_deviation)
np_last_25_mean = last25mean(nose_filter_deviation)




sh_total   = safe_avg(sh_filter_deviation)
ch_total   = safe_avg(ch_filter_deviation)
sp_total   = safe_avg(sp_filter_deviation)
nose_total = safe_avg(nose_filter_deviation)





THRESHOLDS = {
    "shoulder": 9.0,  
    "chest":   10.0,  
    "spine":    8.0,
    "head" : 0.12  
}


DRIFT_BAD = 1.15
DRIFT_GOOD = 0.85


def drift_status(first, last):
    if first == 0:
        return "stable"
    ratio = last / first
    if ratio >= DRIFT_BAD:
        return "worsened"
    elif ratio <= DRIFT_GOOD:
        return "improved"
    else:
        return "stable"

def quality_status(total, threshold):
    return "poor" if total > threshold else "good"

results = {
    "shoulder": {
        "drift": drift_status(sh_first_25_mean, sh_last_25_mean),
        "quality": quality_status(sh_total, THRESHOLDS["shoulder"]),
    },
    "chest": {
        "drift": drift_status(ch_first_25_mean, ch_last_25_mean),
        "quality": quality_status(ch_total, THRESHOLDS["chest"]),
    },
    "spine": {
        "drift": drift_status(sp_first_25_mean, sp_last_25_mean),
        "quality": quality_status(sp_total, THRESHOLDS["spine"]),
    },
    "head": {
        "drift": drift_status(np_first_25_mean, np_last_25_mean),
        "quality": quality_status(nose_total, THRESHOLDS["head"]),
    }
}



feedback = []

if results["head"]["drift"] == "worsened":
    feedback.append("You moved your head more as the interview progressed, which can signal nervousness.")
elif results["head"]["quality"] == "poor":
    feedback.append("Frequent head movement may reduce perceived confidence.")
else:
    feedback.append("Your head movement was controlled and confident.")

if results["shoulder"]["drift"] == "worsened":
    feedback.append("Your shoulders became increasingly uneven, indicating rising tension.")
elif results["shoulder"]["quality"] == "poor":
    feedback.append("Your shoulders were uneven for most of the interview.")
else:
    feedback.append("Your shoulder posture remained stable.")

if results["chest"]["drift"] == "worsened":
    feedback.append("Your chest posture collapsed over time, suggesting reduced confidence.")
elif results["chest"]["quality"] == "good":
    feedback.append("You maintained an open and confident chest posture.")
else:
    feedback.append("Your chest posture needs improvement.")

if results["spine"]["drift"] == "worsened":
    feedback.append("You leaned more as the interview progressed.")
elif results["spine"]["quality"] == "poor":
    feedback.append("Your spine alignment was inconsistent.")
else:
    feedback.append("Your spine posture was upright and stable.")



score = 100

if results["shoulder"]["quality"] == "poor": score -= 15
if results["chest"]["quality"] == "poor":    score -= 25
if results["spine"]["quality"] == "poor":    score -= 20
if results["head"]["quality"] == "poor": score -= 10

if results["shoulder"]["drift"] == "worsened": score -= 10
if results["chest"]["drift"] == "worsened":    score -= 15
if results["spine"]["drift"] == "worsened":    score -= 15
if results["head"]["drift"] == "worsened": score -= 10

score = max(0, score)


print("Posture Score:", score)
for f in feedback:
    print("-", f)
    speak(f)
speak(f"Overall Posture Score is {score} ")
