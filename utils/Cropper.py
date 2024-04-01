import cv2
import logging


logging.basicConfig(filename='croplog.log', format="%(asctime)s %(levelname)-8s [%(name)s]: %(message)s", level= logging.INFO)
logger= logging.getLogger('croplog')

## In case there are very large number of files, it makes it easy to track the progress of the cropping.
logger.info('This file contains logs on cropping of images. The numbers are the index of the loop.')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(gray_image, minNeighbors):
    iteration = 1  # to tackle with images always detecting multiple faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, minNeighbors)
    
    if len(faces) == 0:  # no faces detected
        minNeighbors -= 1
        if minNeighbors == 0:  # reached minimum minNeighbors
            return [[0, 0, gray_image.shape[0], gray_image.shape[1]]]
        return extract_faces(gray_image, minNeighbors)
    
    if len(faces) > 1:  # multiple faces detected
        faces = faces[[0]]  # select only the first face
    
    return faces

if __name__=='__main__':
    print("This file can only be imported!!!")