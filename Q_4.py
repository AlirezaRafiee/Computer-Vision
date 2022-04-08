import cv2
import numpy as np
from scipy.signal import convolve2d

def ESC():
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

def Guassian_Filter(size_filter,sigma):
    Filter = np.zeros((size_filter,size_filter))
    x_center = (size_filter - 1) / 2
    y_center = (size_filter - 1) / 2
    C = (1)/(2*(np.pi)*(sigma ** 2))
    for i in range(size_filter):
        for j in range(size_filter):
            EXP = np.exp(( (i-x_center) ** 2 + (j-y_center) ** 2 )/((-2)*(sigma ** 2)))
            Filter[i,j] = C * EXP
    return Filter

def myEdgeFilter(img0, sigma):
    size_filter = int(2 * np.ceil(3*sigma) + 1) # Filter size
    Filter = Guassian_Filter(size_filter, sigma) # Guassian Filter
    img0 = convolve2d(img0, Filter) # Convolution
    Sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) # Sobel filter 
    Sobel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) # Sobel filter
    Image_dx = convolve2d(img0, Sobel_x) # d(image)/dx
    Image_dy = convolve2d(img0, Sobel_y) # d(image)/dy
    Gradient_magnitude = np.sqrt(np.multiply(Image_dx, Image_dx) + np.multiply(Image_dy, Image_dy))
    Gradient_angle = np.arctan(np.divide(Image_dy, Image_dx))
    Gradient_magnitude -= np.min(Gradient_magnitude) # Scale to [0, 255]
    Gradient_magnitude *= (255/np.max(Gradient_magnitude))
    Gradient_magnitude = Gradient_magnitude.astype('uint8')
    ####### Non maximum suppression
    Gradient_degree = np.zeros(Gradient_angle.shape)
    Gradient_degree[(67.5*np.pi/180) >  Gradient_angle.all() >= (22.5*np.pi/180)] = 45
    Gradient_degree[(112.5*np.pi/180) > Gradient_angle.all() >= (67.5*np.pi/180)] = 90
    Gradient_degree[(157.5*np.pi/180) > Gradient_angle.all() >= (112.5*np.pi/180)] = 135
    ######
    for i in range(1, Gradient_degree.shape[0]-1):
        for j in range(1, Gradient_degree.shape[1]-1):
            if(Gradient_degree[i,j] == 0):
                if((Gradient_magnitude[i-1,j] > Gradient_magnitude[i,j]) and (Gradient_magnitude[i+1,j] > Gradient_magnitude[i,j])):
                    Gradient_magnitude[i,j] = 0
            elif(Gradient_degree[i,j] == 45):
                if((Gradient_magnitude[i-1,j+1] > Gradient_magnitude[i,j]) and (Gradient_magnitude[i+1,j-1] > Gradient_magnitude[i,j])):
                    Gradient_magnitude[i,j] = 0
            elif(Gradient_degree[i,j] == 90):
                if((Gradient_magnitude[i,j+1] > Gradient_magnitude[i,j]) and (Gradient_magnitude[i,j-1] > Gradient_magnitude[i,j])):
                    Gradient_magnitude[i,j] = 0
            elif(Gradient_degree[i,j] == 135):
                if((Gradient_magnitude[i+1,j+1] > Gradient_magnitude[i,j]) and (Gradient_magnitude[i+1,j+1] > Gradient_magnitude[i,j])):
                    Gradient_magnitude[i,j] = 0
    return Gradient_magnitude
img0 = cv2.imread('edge_q.jpg')
print(img0.shape)
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1 =  myEdgeFilter(img0, sigma=1)
cv2.imshow("Question_4", img1)
ESC()