import cv2
def doNothing():
    pass

cv2.namedWindow("Question_2", cv2.WINDOW_AUTOSIZE) #Create window
cv2.createTrackbar('Hue', 'Question_2', 0, 179, doNothing) # Hue trackbar
cv2.createTrackbar('Saturation', 'Question_2', 0, 255, doNothing) # Saturation trackbar
cv2.createTrackbar('Value', 'Question_2', 0, 255, doNothing) # Value trackbar

while(True):
    Image = cv2.imread('flowers.png')
    if cv2.waitKey(1) == 27: # Press esc to close window
        break
    
    Hue = cv2.getTrackbarPos('Hue', "Question_2") # Getting hue value
    Saturation = cv2.getTrackbarPos('Saturation', "Question_2") # Getting saturation value
    Value = cv2.getTrackbarPos('Value', "Question_2") # Getting value value
    image_hsv = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV) # Convert image format
    image_hsv[:, :, 0] = image_hsv[:, :, 0] * (Hue/179) # edit Hue channel
    image_hsv[:, :, 1] = image_hsv[:, :, 1] * (Saturation/255) # edit saturation channel
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * (Value/255) # edit Value channel
    Image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    Image = Image.astype('uint8')
    cv2.imshow("Question_2", Image)
cv2.destroyAllWindows()