import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

a=input('Enter the name of the image to reduce the size: ')
## Read the image
img=plt.imread(a+'.jpg')## Opeinging the image
plt.imshow(img) 
plt.axis('off')
# plt.show()
# print(type(img)) ## the image will be in type of array, as the comuter reads the image in the form of pixels in array form

# print(img.shape) ### 4000 are rows, 6000 are columns, 3 are channels(RGB)
# print(img.size)
## Reshaping the pixels of the image

## Convertig the 3d image into 2d, as it has rows columns and channels as 3d
w,h,d=img.shape
image_array=img.reshape(w*h,d)
# print(image_array.shape)

#Normalizing the pixel value

image_array=image_array/255 ## AS the 255 is the max intencity value of RGB

##As we can't operat on all pixels which are on the total image. WE are taking a portion of pixels in the image and operating on that portion and applying altogether in the whole image.


from sklearn.utils import shuffle
## Fitting model on a small sub sample of hte complete image
image_array_sample= shuffle(image_array,random_state=1)[:1000] ## Taking 1000 pixels as sample or for training
# print(image_array_sample.size)## It would be 3000 as the rgb will be multiplied to it

## Training the image using k_means cluster

KMeans=KMeans(n_clusters=6,random_state=1)
KMeans.fit(image_array_sample)

labels=KMeans.predict(image_array)
c=KMeans.cluster_centers_
# print(c)


#Recreating a original image according to labels and each pixels

def rec_image(c,labels,w,h,d):
    image=np.zeros((w,h,d))  ## w,h,d are created with zeros in the image pixels 
    labels_idx=0  ## label index
# now labeling each pixels according to the limited labels
    for i in range(w):
        for j in range(h):
            image[i][j]=c[labels[labels_idx]]
            labels_idx+=1
    return(image)


## Now we are going to plot the original image and the compressed image

plt.figure(1)
plt.axis('off')
plt.imshow(img)
# plt.show()
plt.figure(2)
plt.axis('off')
plt.title('reduced')
plt.imshow(rec_image(c,labels,w,h,d))
# plt.show()
b=input('Enter the name you want to give to compressed image: ')
plt.savefig(b+'.png')
