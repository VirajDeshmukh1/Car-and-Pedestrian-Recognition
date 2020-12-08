import cv2

#Out Image
img_file = 'CarImage.jpeg'
video = cv2.VideoCapture('PedestriansCompilation.mp4')
# video = cv2.VideoCapture('video.mp4')
# video = cv2.VideoCapture('tesla.mp4')
# video = cv2.VideoCapture('tesla1.mp4')
# video = cv2.VideoCapture('pedestrian.mp4')



#pre-trained car and pedestrain classifier
car_tracker_file = 'car_detector.xml'
pedestrain_tracker_file = 'haarcascade_fullbody.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)

#run forever until car stops 
while True:
    #read the current frame
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedetrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around cars
    for (x, y, w, h) in cars :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    
    #draw rectangles around pedestrians
    for (x, y, w, h) in pedestrians :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    
    #Display the image with the faces spotted
    cv2.imshow('Self Driving Car', frame)

    #Dont autoclose (wait for key prress)
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#release the videoCapture object
video.release()

"""

#create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# #draw rectangles around cars
for (x, y, w, h) in cars :
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# #display the image with the faces spotted
cv2.imshow('Clever Programmer Car Detector',black_n_white)

#dont autoclose (wait here in the code and listen for a key press)
cv2.waitKey()
"""

print("Completed")
