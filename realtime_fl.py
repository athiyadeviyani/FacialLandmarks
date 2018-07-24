from imutils import face_utils
import imutils 
import dlib 
import cv2

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

camera = cv2.VideoCapture(0) # capture images from default input device
count = 0 # initialize the count for the number of faces 

while True: 

    ret, image = camera.read()
    image = cv2.flip(image, 1) # mirror the image (flip vertically)
    image = imutils.resize(image, width = 800, height = 600) # (set the size of the frame)
    faces = detector(image, 0)
    
    for face in faces:
        faceshape = predictor(image, face)
        faceshape = face_utils.shape_to_np(faceshape) # convert to numpy array to be iterable
        for (x,y) in faceshape:
            cv2.circle(image, (x,y), 1, (255, 255, 255), 2) # display white dots in their respective locations (facial landmarks)
    
    cv2.putText(image,"Number of faces found: " + str(len(faces)), (180, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1) 
    cv2.imshow("Image", image) # display the image frame
    
    key = cv2.waitKey(1) & 0xff
    if key == 27: # press ESC to close
        break
