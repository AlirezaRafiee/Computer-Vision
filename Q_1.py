import cv2
import numpy as np
from scipy.signal import convolve2d

def ESC(): ## Press ESC to close window
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
def Guassian_Filter(size_filter,sigma): ## Gaussian filter used to smooth image before downsampling
    Filter = np.zeros((size_filter, size_filter))
    x_center = (size_filter - 1) / 2
    y_center = (size_filter - 1) / 2
    C = (1)/(2*(np.pi)*(sigma ** 2))
    for i in range(size_filter):
        for j in range(size_filter):
            EXP = np.exp(( (i-x_center) ** 2 + (j-y_center) ** 2 )/((-1)*(sigma ** 2)))
            Filter[i,j] = C * EXP
    return Filter

# Part_1 

Image_1 = cv2.imread('parrots.png')

size_filter = 3 #Create gaussian filter
sigma = 0.5
Filter = Guassian_Filter(size_filter,sigma)

temp = convolve2d(Image_1[:,:,0], Filter) # filtering image for its 3 channels
Image_1_filtered = np.zeros((temp.shape[0], temp.shape[1], 3))
Image_1_filtered = Image_1_filtered.astype('uint8')
Image_1_filtered[:,:,0] = temp
Image_1_filtered[:,:,1] = convolve2d(Image_1[:,:,1], Filter)
Image_1_filtered[:,:,2] = convolve2d(Image_1[:,:,2], Filter)

Image_1_scaled = Image_1_filtered[0::2, 0::2, :] # downsampling the image
cv2.imshow("Part_1: rescale the image with scale=0.5", Image_1_scaled)
ESC()
a = Image_1_scaled

## Part 2
gray_scale = np.stack((cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),)*3, axis=-1) # Create gray-scale image
binary_image = 255*(gray_scale > 73) # create binary image with thr = 73
horizontal_image = a[a.shape[0]::-1,:,:] # horizontal mirror of image a

Image_part_2 = np.zeros((2*a.shape[0], 2*a.shape[1], 3)) #Concatenating 4 images
Image_part_2 = Image_part_2.astype('uint8')
Image_part_2[0:a.shape[0], 0:a.shape[1], :] = a
Image_part_2[0:a.shape[0], a.shape[1]:, :] = gray_scale
Image_part_2[a.shape[0]:, 0:a.shape[1], :] = binary_image
Image_part_2[a.shape[0]:, a.shape[1]:, :] = horizontal_image

cv2.imshow("Part_2: Concatenating 4 images", Image_part_2)
ESC()

## Part_3
start_point = (a.shape[1],0) #draw the horizontal line
end_point = (a.shape[1], 2*a.shape[0])
color = (255,255,255)
thickness = 9
Image_lined = cv2.line(Image_part_2, start_point, end_point, color, thickness)
start_point = (0,a.shape[0]) #draw the vertical line
end_point = (2*a.shape[1], a.shape[0])
color = (255,255,255)
thickness = 9
Image_lined = cv2.line(Image_lined, start_point, end_point, color, thickness)
# Put text on the image
cv2.putText(img=Image_lined, text='Scaled 0.5', org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)
cv2.putText(img=Image_lined, text='Binary image', org=(10, a.shape[0]+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)
cv2.putText(img=Image_lined, text='Gray Image', org=(10+a.shape[1], 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)
cv2.putText(img=Image_lined, text='Horizontal mirror', org=(10+a.shape[1], a.shape[0]+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)

cv2.imshow("Part_3: Putting text on image", Image_lined)
ESC()
cv2.imwrite('parrots_out.png',Image_lined)