import os
import sys
import numpy as np
import skimage.io
import json


class Annotations:
    
    def __init__(self):
        self.img_ids = []
        self.img_info = []
        

    def load(self, dataset_dir):

        
        # Define the names and the corrwspon
        self.class_info = [{"id": 0, "name": "BG"},
                           {"id": 1, "name": "intergranular"},
                           {"id": 2, "name": "transgranular"}]
              
               
        # load json file, created by VGG image annotator
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # keep only the values
        annotations = list(annotations.values())  

        # Keep only images that have been annotated
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # The x,y coordinates of each corner of a polygon are used to create masks for the different annotated areas.
            # These coordinates are stored in the shape_attributes
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]
            # The image dimensions are also needed for the creation of the mask
            img_path = os.path.join(dataset_dir, a['filename'])
            img = skimage.io.imread(img_path)
            height, width = img.shape[:2]
            
            self.img_info.append({ "id": a['filename'],
                          "source": "fracture",
                          "path": img_path,
                          "width":width,
                          "height":height,
                          "polygons":polygons,
                          "names":names})
    def organize(self):
        
        # Define the class parameters needed.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_imgs = len(self.img_info)
        self.img_ids = np.arange(self.num_imgs)


                    
                    
    def load_img(self, img_id):
        
        # Load image
        img = skimage.io.imread(self.img_info[img_id]['path'])
        # If grayscale. Convert to RGB.
        if img.ndim != 3:
            img = skimage.color.gray2rgb(img)
        # If has an alpha channel, remove it.
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img
    
    
    def load_mask(self, img_id):
        """This function returns the masks for each image
 
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        
        info = self.img_info[img_id]
        class_names = info["names"]
        # we create an array of masks with dimensions equal to the image dimensions and zeros everywhere except
        # for the annotated areas. This mask array holds one mask for each annotation.
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # The index i is the annotation number and p holds the x,y coordinates of the polygon corners
            # Each mask is an array with zeros everywhere except from the pixels that belong to annotated areas
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Assign class_ids
        class_ids = np.zeros([len(info["polygons"])])
        # In the fracture dataset, pictures are labeled with name 'i' and 't' representing intergranular and transgranular fracture.
        for j, p in enumerate(class_names):
        #"name" is the attributes name decided when performing the annotation with VGG anotator, etc. 'region_attributes': {name:'n'}
            if p['name'] == 'i':
                class_ids[j] = 1
            elif p['name'] == 't':
                class_ids[j] = 2

        class_ids = class_ids.astype(int)
           
        return mask.astype(np.bool), class_ids
