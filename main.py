import cv2
import numpy as np
import face_recognition
#step 1
imgelon= face_recognition.load_image_file('face_pic/ellon1.jpg')
imgelon= cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB) #to convert the image into rgb
imgtest= face_recognition.load_image_file('face_pic/bill.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB) #to convert the image into rgb

#step 2 -finding faces in image and finding their encodings
faceloc= face_recognition.face_locations(imgelon)[0]#since sending only single image we will take first element of this
encodeelon= face_recognition.face_encodings(imgelon)[0]#print(faceloc)# prints four num. of four corners
cv2.rectangle(imgelon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)#2 is thickness. other are colors

faceloctest= face_recognition.face_locations(imgtest)[0]
encodeelontest= face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,0),2)

#step 3 -comparing the faces and finding the distance between them
results = face_recognition.compare_faces([encodeelon],encodeelontest)
#to check the similarity btw two images. lower the dis more similarity present.
facedis=face_recognition.face_distance([encodeelon],encodeelontest)
print(results)
print(facedis)
cv2.putText(imgtest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
cv2.imshow('elon musk',imgelon)
cv2.imshow('elon test',imgtest)
cv2.waitKey(0)