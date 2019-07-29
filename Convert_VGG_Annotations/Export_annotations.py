"""
The main objective of this code is to convert the annotations exported in a json file by the 
VGG Image Annotator into a format that the main segmentation program can use.
The format that the segmentation code use to annotate the images is an image where each pixel has a 
value equal to the class id that it belongs. 
"""




import os
import sys
import numpy as np
import cv2
import scipy.misc

from Annotate import Annotations

# Root directory of the project
ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)


# Annotations directory
ANNOT_DIR = os.path.join(ROOT_DIR, "fracture_images\\val")


# Create a annotation class object
fracture_annotations = Annotations()

# Load all the information from the json file produced by the VGG Image Annotator
fracture_annotations.load(ANNOT_DIR)

# organize the information and store it at the annotation object parameters
fracture_annotations.organize()

# Print the number of images and the existing classes
print("Image Count: {}".format(len(fracture_annotations.img_ids)))
print("Class Count: {}".format(fracture_annotations.num_classes))
for i, info in enumerate(fracture_annotations.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
    
    
img_ids = fracture_annotations.img_ids

for img_id in img_ids:
    # Image information
    info = fracture_annotations.img_info[img_id]
    # Load the image
    img = fracture_annotations.load_img(img_id) 
    # Store the coordinates of the polygons of each annotation in the mask and the corresponding class_ids 
    mask, class_ids = fracture_annotations.load_mask(img_id)
    class_names = fracture_annotations.class_names
    
    # Pull masks of instances belonging to the same class.
    m1 = mask[:, :, np.where(class_ids == 1)[0]]
    m1 = np.sum(m1*255, -1)
    m2 = mask[:, :, np.where(class_ids == 2)[0]]
    m2 = np.sum(m2*80, -1)
    t = m1 + m2 
    t = t.astype(np.uint8)
    t[t<=20] = 0
    t[(t>20) & (t<200)] = 1
    t[t>=200] = 2
    
    # Export the annotated images in the format that the segmentation code requires.
    img_dir = os.path.join(ROOT_DIR, "fracture_images\\annotations_val")
    img_name = info["id"][:-4] + ".png"
    out_dir = os.path.join(img_dir, img_name)
    
    cv2.imwrite(out_dir, t.astype(np.uint8))    
    print("Image {} is converted !!!".format(info["id"]))
