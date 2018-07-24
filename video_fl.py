from imutils import face_utils
import imutils 
import dlib 
import cv2

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

camera = cv2.VideoCapture("avengers.mp4")
# video courtesy of youtube
# video from http://www.youtube.com/watch?v=L3ymBk6Vb04

count = 0 # Initialize the count for the number of faces

while True: 

    ret, vid = camera.read()
    vid = imutils.resize(vid, width = 1000, height = 800) # (set the size of the frame)
    faces = detector(vid, 0)

    for face in faces:
        faceshape = predictor(vid, face)
        faceshape = face_utils.shape_to_np(faceshape) # convert to numpy array
        for (x,y) in faceshape:
            cv2.circle(vid, (x,y), 1, (255, 255, 255), -1) # display white dots in their respective locations (facial landmarks)
    
    cv2.putText(vid,"Number of faces found: " + str(len(faces)), (280, 550), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
    cv2.imshow("Video", vid) # display the video frame
    
    key = cv2.waitKey(1) & 0xff
    if key == 27: # press ESC to close
        break
