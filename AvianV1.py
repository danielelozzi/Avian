#!/usr/bin/env python
# coding: utf-8
def scalable_cell_cutter_40(file_name,folder_name):
    magnification = 40
    
    import skimage
    import pandas as pd
    import time
    import matplotlib
    import scipy
    #import sklearn
    import os
    import math
    from skimage.segmentation import quickshift as qs
    from skimage.segmentation import slic
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_fill_holes
    from skimage.exposure import histogram
    from skimage.color import rgb2hsv, rgb2hed, rgb2gray
    from skimage.filters import threshold_mean,sobel,roberts,threshold_otsu
    from skimage.feature import canny
    #from scipy import ndimage as ndi
    from skimage.measure import find_contours
    import numpy as np
    import matplotlib.pyplot as plt 
    #import urllib.request
    from PIL import Image
    import pandas as pd
    #import cv2
    #from tensorflow.keras.preprocessing import image
    from pillow_heif import register_heif_opener
    
    register_heif_opener()

    def create_circular_mask(h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def inverte(image,binary=False):
        if binary== False:
            image = (255-image)
            image = abs(image)
            return image
        if binary==True:
            image = (1-image)
            image = abs(image)
            return image

    def calculate_perimeter(a,b):
        perimeter = math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )
        return perimeter

    cell = Image.open(file_name)
    cell = np.array(cell)
    
    if (cell.shape[0] > 2500) and (cell.shape[0] < 5000) and (cell.shape[1] > 2500) and (cell.shape[0] < 5000):
        t1 = time.time()
        # exclusion alpha channel
        if cell.shape[-1] == 4:
            cell = cell[:,:,0:3]

        # divide into channels
        red = cell[:,:,0]
        green = cell[:,:,1]
        blue = cell[:,:,2]


        multi_threshold_0 = skimage.filters.threshold_multiotsu(red,classes=6)

        cell_exposed = skimage.exposure.rescale_intensity(cell,in_range=(multi_threshold_0[0],multi_threshold_0[-1]))

        # divide into channels
        red_exposed = cell_exposed[:,:,0]
        green_exposed = cell_exposed[:,:,1]
        blue_exposed = cell_exposed[:,:,2]

        multi_threshold_1 = skimage.filters.threshold_multiotsu(red_exposed,classes=6)

        binary_red_exposed = red_exposed < multi_threshold_1[0]

        label_bg = skimage.measure.label(binary_red_exposed,background=True)
        label_gb_table = skimage.measure.regionprops_table(label_bg, properties=('centroid',
                                                         'orientation',
                                                         'axis_major_length',
                                                         'axis_minor_length','bbox','image','label','area_bbox','area','eccentricity'))
        label_gb_dataframe = pd.DataFrame(label_gb_table)
        label_gb_dataframe = label_gb_dataframe.loc[(label_gb_dataframe.area > 10000)] 
        label_gb_dataframe = label_gb_dataframe.sort_values('eccentricity')
        
        mask = create_circular_mask(binary_red_exposed.shape[0],binary_red_exposed.shape[1], center=(label_gb_dataframe['centroid-1'].iat[0],label_gb_dataframe['centroid-0'].iat[0]), radius=label_gb_dataframe['axis_major_length'].iat[0]/2)
        
        mask = np.stack((mask,mask,mask),axis=-1)
        cell_exposed = cell_exposed*mask

        xmin = label_gb_dataframe['bbox-0'].iat[0]
        xmax = label_gb_dataframe['bbox-2'].iat[0]
        ymin = label_gb_dataframe['bbox-1'].iat[0]
        ymax = label_gb_dataframe['bbox-3'].iat[0]

        cell_exposed = cell_exposed[xmin:xmax,ymin:ymax]

        red_exposed = cell_exposed[:,:,0]

        red_equalized = skimage.exposure.equalize_adapthist(red_exposed,kernel_size=100,clip_limit=0.5)

        multi_threshold_2 = skimage.filters.threshold_multiotsu(red_equalized,classes=6)

        binary_red_exposed = red_equalized < multi_threshold_2[0]

        binary_red_exposed = binary_red_exposed*1
        blur = skimage.filters.gaussian(binary_red_exposed,sigma=2)
        ero = skimage.morphology.erosion(blur)
        for i in range(3): #20
            ero = skimage.morphology.erosion(ero)
        for i in range(5): #15
            ero = skimage.morphology.dilation(ero)
        padding = min(int(cell_exposed.shape[0]/10),int(cell_exposed.shape[1]/10))       

        r=np.pad(cell_exposed[:,:,0],padding)
        g=np.pad(cell_exposed[:,:,1],padding)
        b=np.pad(cell_exposed[:,:,2],padding)
        cell_exposed = np.stack((r,g,b),axis=-1)

        #ero = np.pad(ero,min(int(cell.shape[0]/10),int(cell.shape[1]/10)),constant_values=1,mode='constant')
        #edges = canny(ero)
        ero = np.pad(ero,padding)
        contours = skimage.measure.find_contours(ero,fully_connected='high')

        final_contour = list()
        for contour in contours:
            #if (len(contour) < 1000) and (len(contour) > 200):
            xlen1 = np.take(contour,0,axis=1).max() - np.take(contour,0,axis=1).min()
            ylen1 = np.take(contour,1,axis=1).max() - np.take(contour,1,axis=1).min()
            if (calculate_perimeter(xlen1,ylen1) > 50) and (calculate_perimeter(xlen1,ylen1) <500):
                final_contour.append(contour)

        image = np.zeros(cell_exposed.shape[0:2])
        for contour in final_contour:
            for pixel in contour:
                image[int(pixel[0]),int(pixel[1])] = 1

        image_mask = scipy.ndimage.binary_fill_holes(image)
        #image_mask = np.pad(image_mask,padding,constant_values=0,mode='constant')

        plt.figure(figsize=(12,12))
        plt.imshow(image_mask)

        labels = skimage.measure.label(image_mask*1)
        props = skimage.measure.regionprops_table(labels, properties=('centroid',
                                                         'orientation',
                                                         'axis_major_length',
                                                         'axis_minor_length','bbox','image','label','area_bbox','area'))
        #props = pd.DataFrame(props)
        image_label_overlay = skimage.color.label2rgb(labels, image=image_mask, bg_label=0)
        centroids = list()
        fig, ax = plt.subplots(figsize=(50,50))
        ax.imshow(image_label_overlay)
        ax.imshow(cell_exposed)

        for region in skimage.measure.regionprops(labels):
            # take regions with large enough areas
            if region.area >= 50:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
                centroids.append([region.centroid,region.axis_major_length])

        #folder_name = file_name.rsplit('.')[0]
        if folder_name not in os.listdir('./'):
            os.mkdir(folder_name)

        plt.savefig('./'+folder_name+'/bounding_box.jpg')


        n=0

        for c in centroids: 
            xmin = int(c[0][0]-c[1])
            xmax = int(c[0][0]+c[1])
            ymin = int(c[0][1]-c[1])
            ymax = int(c[0][1]+c[1])


            im = Image.fromarray(cell_exposed[xmin:xmax,ymin:ymax])
            im = np.array(im)

            mask = create_circular_mask(im.shape[0],im.shape[1], center=None, radius=c[1])
            mask = mask*1
            im[:,:,0] = im[:,:,0]*mask
            im[:,:,1] = im[:,:,1]*mask
            im[:,:,2] = im[:,:,2]*mask

            name = './'+folder_name+'/cell_'+str(n)+folder_name+'.jpg'
            plt.imsave(name,im)
            n+=1

        print('cells counted:',len(centroids))
        with open('./'+folder_name+'/n_cells.txt','w') as f:
            f.write(str(len(centroids)))

        t2 = time.time()
        print('computational time = ',t2-t1)
        computational_time = t2-t1
        with open('./'+folder_name+'/time.txt','w') as f:
            f.write(str(computational_time))


    
    else:
        print('image shape is not compatible')