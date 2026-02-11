import cv2 #import opencv librarie
import os

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

while True:
    success , img = cap.read()  #infinite loop reads from camera
    
    img_resized = cv2.resize(img, (640, 480)) #Resizes the camera frame to 640x480 pixels, regardless of the original camera resolution.
    imgBackground[162:162+480, 55:55+640] = img_resized 
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    """Places the resized frame onto a larger background image (imgBackground).
    [162:162+480, 55:55+640] is a slice of the background where the frame should appear:
    Rows: 162 → 162+480 (vertical placement)
    Columns: 55 → 55+640 (horizontal placement)"""

    #cv2.imshow("Webcam", img)  #Shows the cam 
    cv2.imshow("Face Attendance", imgBackground)  #shows the attendance sys 
    cv2.waitKey(1)       
                            
