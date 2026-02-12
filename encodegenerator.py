import cv2
import face_recognition
import pickle
import os

#importing the student img 
folderPath = 'Images'
PathList = os.listdir(folderPath)
#print(PathList)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    #print(os.path.splitext(path)[0]) #gets the id
    studentIds.append(os.path.splitext(path)[0])#gets the id from the img name 
print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] #Detects the face in the image and returns a 128-dimensional vector representing that face.
        encodeList.append(encode) #Stores the face encoding for later use (matching faces).
        
    return encodeList

print("Encoding starts...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,studentIds]
# print(encodeListKnown)
print("Encoding complete")


file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")