import cv2

#Loading some pretrained data in face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect face in
#img = cv2.imread('RDJ.png')
webcam = cv2.VideoCapture('videoplayback.mp4')

#Iterate forever over frames
while True:

    #Read current frame
    success_frame_read, frame = webcam.read()
    
    #Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv2.imshow('face-detector', frame)
    key = cv2.waitKey(1)

    ### Press q to Stop the loop
    if key==81 or key==113:
        break

### Release the VideoCapture object
webcam.release()

print("code complete")


    


# #Converting to grayscale
# grayscaled_img = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)

# #Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# #Draw rectangles around the faces
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(webcam, (x, y), (x+w, y+h), (0, 0, 255), 3)

# #print(face_coordinates)


# # Display image with the faces
# cv2.imshow('face-detector', webcam)
# cv2.waitKey()

# print("Code Complete")