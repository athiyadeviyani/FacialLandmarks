# Plots 68 facial landmarks in a static image

from imutils import face_utils
import imutils 
import dlib 
import cv2

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

count = 0 # initialize the count for the number of faces

while True: 

    image = dlib.load_rgb_image("friends.jpg") # loads the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV reads BGR instead of RGB, so we need to convert the colors
    image = imutils.resize(image, width = 800, height = 600)
    faces = detector(image, 1)

    for face in faces:
        faceshape = predictor(image, face)
        faceshape = face_utils.shape_to_np(faceshape) # convert to numpy array
        for (x,y) in faceshape:
            cv2.circle(image, (x,y), 1, (255, 255, 255), -1) # display white dots in their respective locations (facial landmarks)
    
    cv2.putText(image,"Number of faces found: " + str(len(faces)), (180, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
    cv2.imshow("Image", image) # display the image and the landmarks
    
    key = cv2.waitKey(1) & 0xff
    if key == 27: # press ESC to close
        break
