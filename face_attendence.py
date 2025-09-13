import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'image_attendence'
#create following lists
images = []
classNames = []
list =[]
list = os.listdir(path)# list contains the name of all images in image_attendence directrary
print(list)
for cl in list:
    curimg = cv2.imread(f'{path}/{cl}') # here f is used to refer to the variables in the string.
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])# this removes .jpg from the name of the image.
print(classNames)
def findencodings(images):
    encoding_list = [] #create an encoding list
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#step 1
        encodeimg = face_recognition.face_encodings(img)[0]#step 2
        encoding_list.append(encodeimg) #appends each image encodings to this list
    return encoding_list

def markattendence(name):
    with open('Attendace.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist =[]
        for line in mydatalist:
            entry = line.split(',')#contains both name and time
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtstring}')



encodelistknown = findencodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)#to capture real time video

while True:
    success, img = cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs= cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facecurrframe = face_recognition.face_locations(imgs)
    encodecurrframe = face_recognition.face_encodings(imgs, facecurrframe)

    for encodeface,faceloc in zip(encodecurrframe,facecurrframe): #use zip becacuse we want both in same loop.
        matches= face_recognition.compare_faces(encodelistknown,encodeface)
        facedis= face_recognition.face_distance(encodelistknown,encodeface)
        #print(facedis)
        matchind = np.argmin(facedis)

        if matches[matchind]:
            name = classNames[matchind]
            #print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(255, 0, 0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (255, 0, 0), cv2.FILLED)# for face recognized rectangle
            cv2.putText(img, name,(x1-6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)#for a name box
            markattendence(name)
    cv2.imshow('webcame', img)
    cv2.waitKey(1)




