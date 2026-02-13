import cv2 #import opencv librarie
import os
import pickle
import face_recognition
import numpy as np

cap = cv2.VideoCapture(1) #This line opens a video capture object
cap.set(3,640) #width
cap.set(4,480)  #height

imgBackground = cv2.imread('Resources/background.png')


#importing the mode img into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    
#print(len(imgModeList))

#Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file) 
file.close()
encodeListKnown,studentIds = encodeListKnownWithIds
#print(studentIds)
print("Encode File Loaded ...")
while True:
    success , img = cap.read()  #infinite loop reads from camera
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    img_resized = cv2.resize(img, (640, 480)) #Resizes the camera frame to 640x480 pixels, regardless of the original camera resolution.
    imgBackground[162:162+480, 55:55+640] = img_resized 
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    
    
    for encodeFace , faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print("Matches",matches)
        print("Face Distance",faceDis)
        
        matchindex = np.argmin(faceDis)
        print("Match Index", matchindex)
        if matches[matchindex]:
            top, right, bottom, left = faceLoc
            top = int(top * 4)
            right = int(right * 4)
            bottom = int(bottom * 4)
            left = int(left * 4)

            # Because we resized to 640x480 for background, compute scale ratio
            h_ratio = 480 / img.shape[0]
            w_ratio = 640 / img.shape[1]

            top = int(top * h_ratio)
            bottom = int(bottom * h_ratio)
            left = int(left * w_ratio)
            right = int(right * w_ratio)

            # Draw rectangle with offset
            cv2.rectangle(imgBackground,
                          (left + 55, top + 162),
                          (right + 55, bottom + 162),
                          (0, 255, 0), 2)

            # Optional: add student ID
            cv2.putText(imgBackground,
                        str(studentIds[matchindex]),
                        (left + 55, top + 162 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

            print("Known Face Detected")
            print(studentIds[matchindex])
            
        else :
            print("Not deticted yet")
    """Places the resized frame onto a larger background image (imgBackground).
    [162:162+480, 55:55+640] is a slice of the background where the frame should appear:
    Rows: 162 → 162+480 (vertical placement)
    Columns: 55 → 55+640 (horizontal placement)"""

    #cv2.imshow("Webcam", img)  #Shows the cam 
    cv2.imshow("Face Attendance", imgBackground)  #shows the attendance sys 
    cv2.waitKey(1)       
                            
