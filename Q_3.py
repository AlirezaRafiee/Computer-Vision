import numpy as np
import cv2

def ESC(): ## Press ESC to close window
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        
def myImageFilter(img0, h):
    size_x = h.shape[0] # size x filter
    size_y = h.shape[1] # size y filter
    pad_x = int((size_x-1)/2) 
    pad_y = int((size_y-1)/2)
    Dtype = img0.dtype
    Output_image = np.zeros((img0.shape[0], img0.shape[1]))
    Image_paded = np.pad(img0, ((pad_x, pad_x),(pad_y, pad_y)), 'edge') # padding image with edge values
    for i in range(pad_x, Image_paded.shape[0]-pad_x):
        for j in range(pad_y, Image_paded.shape[1]-pad_y):
            temp = Image_paded[i-pad_x:i+pad_x+1, j-pad_y:j+pad_y+1]
            Output_image[i-pad_x,j-pad_y] = np.sum(np.multiply(temp, h))
    Output_image = Output_image.astype(Dtype)
    return Output_image

h = np.ones((5, 5))
h /= np.sum(h)
Image = cv2.imread('flowers.png')
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
print(f'shape of image before convolution: {Image.shape}')
Image_filtered = myImageFilter(Image, h)
print(f'shape of image after convolution: {Image_filtered.shape}')
cv2.imshow("Question_3_lowpass filter", Image_filtered)
ESC()
h = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
Image_filtered = myImageFilter(Image, h)
cv2.imshow("Question_3_highpass filter", Image_filtered)
ESC()

