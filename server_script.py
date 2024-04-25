import dlib
import cv2
import numpy as np
import os
from PIL import Image
from Tele_msg import send_msg
from datetime import datetime
from flask import Flask, request, jsonify

pwd = os.path.dirname(__file__) #getting current directory

def write_log(message): #function to write log

    log_file_path = pwd + '/log_file.txt'
    max_lines = 1000
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Concatenate timestamp and message
    log_entry = f'{timestamp} -> {message}'
    # Write the message to the log file
    with open(log_file_path, 'a') as file:
        file.write(log_entry + '\n')

    # Check if the number of lines exceeds the maximum limit
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        if num_lines > max_lines:
            # Delete the oldest line (top line)
            lines_to_keep = lines[-max_lines:]
            with open(log_file_path, 'w') as file:
                file.writelines(lines_to_keep)



# Load pre-trained face detector and facial recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(pwd + "/shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1(pwd + "/dlib_face_recognition_resnet_model_v1.dat")
write_log('Recognition Models Loaded..')

def find_faces(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downsample the image
    img_downsampled = cv2.resize(gray, None, fx=0.5, fy=0.5)

    # Detect faces in the downsampled image
    faces = detector(img_downsampled, 1)

    # Upscale the coordinates of detected faces
    faces_upscaled = [dlib.rectangle(int(face.left() * 2), int(face.top() * 2),
                                     int(face.right() * 2), int(face.bottom() * 2))
                      for face in faces]

    return faces_upscaled

#DEFINING MODELS AND DATASET
path = pwd + '/Dataset'
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
images = [cv2.imread(os.path.join(path, cl)) for cl in os.listdir(path)]
write_log('Familiar Face database loaded')

# Encode the known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = find_faces(imgRGB)
        if faces:
            landmarks = shape_predictor(imgRGB, faces[0])
            encode = np.array(face_recognizer.compute_face_descriptor(imgRGB, landmarks))
            encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images) #obtaining the encodings of known database
write_log('Face Encodings Created')


#function to mark the faces
def mark_faces(img,color,name,face): #function to mark faces after recognition
    # Draw rectangle around the face and display the name
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    #x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    #cv2.rectangle(img, (x1, y2 - 35), (x2 + 60, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save both marked images
    output_path = pwd + "/output_img/output_image.jpg"
    cv2.imwrite(output_path, img)
    write_log('Marked face image saved')

    return output_path

# Main loop
def face_detection(img):
    #img_path = input("Enter image path:")
    #img = cv2.imread(img_path)
    if img is None:
        write_log("Invalid image path. Please try again.")
        #print("Invalid image path. Please try again.")

    #imgS = cv2.resize(img,(0, 0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the input image
    facesCurFrame = find_faces(imgS)

    face_len = len(facesCurFrame)
    if face_len <= 0:
        write_log('No faces detected in the image ')
    else:
        write_log(f'No of faces detected : {face_len}')

    for face in facesCurFrame:
        #print(f'Face object :{face}')
        # Extract face encodings
        landmarks = shape_predictor(imgS, face)
        encodeFace = np.array(face_recognizer.compute_face_descriptor(imgS, landmarks))

        # Perform face recognition on each detected face
        matches = [np.linalg.norm(encodeFace - encode) < 0.4 for encode in encodeListKnown]
        #set threshold for face detection here( reverse order, less = more accurate)

        if True in matches:
            matchIndex = matches.index(True)
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)  # Green for known faces
            path = mark_faces(img,color,name,face) #marks the face boundaries
            write_log("Known face detected - no alerts send!")
            continue

        else:
            name = 'Unknown'
            color = (0, 0, 255)  # Red for unknown faces
            path = mark_faces(img,color,name,face) #marks the face boundaries
            send_msg(path) #sends telegram message
            write_log("Telegram Alert Send for unknown face")
            continue

#-- MAIN FUNTION --

app = Flask(__name__)

# Define a route to receive images
@app.route('/', methods=['POST'])
def process_image():
    # Receive the image from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    write_log('image file recieved, Initiating Face Recogniton..')

    face_detection(image) #initialize face detection

    # For demonstration, just return a success message
    return jsonify({'message': 'Image processed successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
