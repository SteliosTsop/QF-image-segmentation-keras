import argparse, os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import Utils
import glob
import cv2
from PIL import Image, ImageDraw
import numpy as np



# Parse command line arguments and assign them to parameters
parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

n_classes = args.n_classes
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
trained_weights = args.save_weights_path

# Initialize a model and load pre-trained weights
keras_model = Utils.VGG16_Unet(n_classes, False, input_height=input_height, input_width=input_width)
keras_model.load_weights(trained_weights)


# Compile the model
keras_model.compile(loss='categorical_crossentropy',optimizer= 'adam', metrics=['accuracy'])
# keras_model.summary() # Display the CNN architecture


# Define output dimensions
output_height = keras_model.outputHeight
output_width = keras_model.outputWidth

# Define the image paths
images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.tiff")
images.sort()

# Colors for each label class
colors = [(0,0,0),(0,250,0),(250,0,0)]


for imgName in images:
    outName = imgName[:-4] + "_pred.jpg"
    X = Utils.get_images(imgName, input_width, input_height)
    pr = keras_model.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax( axis=2 )
    img = cv2.imread(imgName, 1)
    seg_img = cv2.resize(img, (output_width, output_height))
    for c in range(1,n_classes):
        seg_img[:, :, 0] = np.where(pr[:, :] == c, seg_img[:, :, 0]*0.65 + 0.35*colors[c][0], seg_img[:, :, 0])
        seg_img[:, :, 1] = np.where(pr[:, :] == c, seg_img[:, :, 1]*0.65 + 0.35*colors[c][1], seg_img[:, :, 1])
        seg_img[:, :, 2] = np.where(pr[:, :] == c, seg_img[:, :, 2]*0.65 + 0.35*colors[c][2], seg_img[:, :, 2])
    
    
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)    

    print("image {} is done!!!".format(outName))
    
