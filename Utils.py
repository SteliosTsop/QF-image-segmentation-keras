from keras.models import *
from keras.layers import *
import os
import numpy as np
import cv2
import glob
import itertools,random


# define the path to load the VGG16 weights
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG16_Weights_path = file_path + "/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"




def VGG16_Unet( n_classes , load_vgg, input_height, input_width, batch_norm = False ):

    # This function defines the architecture of the U-net.
    # It is used to initialize the keras model.
    
    
    assert input_height%32 == 0
    assert input_width%32 == 0

    img_input = Input(shape=(input_height,input_width,3))
    
    
    # ENCODER = VGG16 (without the top layers) 
    
    # Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(c1)

    # Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(c2)

    # Block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(c3)

    # Block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(c4)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(c4)

    # Block 5
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(c5)
    p5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(c5)

    # In case the training is not continued from a previous pre-trained state,
    # it is needed to load the pre-trained weights for the encoder part
    if load_vgg:
        vgg = Model(img_input, p5)
        vgg.load_weights(VGG16_Weights_path, by_name=True)
        print("Loading VGG !!!")

    # DECODER
    c6 = Conv2D(1024, (3, 3), padding='same')(p5)

    u6 = UpSampling2D((2, 2))(c6)
    u6 = concatenate([u6, c5], axis=3) # concatenate the weights from the corresponding encoder layer
    c7 = Conv2D(1024, (3, 3), padding='same')(u6)
    c7 = Conv2D(1024, (3, 3), padding='same')(c7)

    u7 = UpSampling2D((2, 2))(c7)
    u7 = concatenate([u7, c4], axis=3) # concatenate the weights from the corresponding encoder layer
    c8 = Conv2D(512, (3, 3), padding='same')(u7)
    c8 = Conv2D(512, (3, 3), padding='same')(c8)

    u8 = UpSampling2D((2, 2))(c8)
    u8 = concatenate([u8, c3], axis=3)
    c9 = Conv2D(256, (3, 3), padding='same')(u8)
    c9 = Conv2D(256, (3, 3), padding='same')(c9)

    u9 = UpSampling2D((2, 2))(c9)
    u9 = concatenate([u9, c2], axis=3) # concatenate the weights from the corresponding encoder layer
    c10 = Conv2D(128, (3, 3), padding='same')(u9)
    c10 = Conv2D(128, (3, 3), padding='same')(c10)

    u10 = UpSampling2D((2, 2))(c10)
    u10 = concatenate([u10, c1], axis=3)
    c11 = Conv2D(128, (3, 3), padding='same')(u10)
    c11 = Conv2D(128, (3, 3), padding='same')(c11)

    o = Conv2D(n_classes, (3, 3), padding='same')(c11)

    o_shape = Model(img_input, o).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((outputHeight * outputWidth, n_classes)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model

def get_images(path, width, height, imgNorm="sub_mean"):
    
    # Load images    
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width , height ))
    img = img.astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
        
    return img


def get_labels(path, nClasses, width, height):
    
    # Load labels
    seg_labels = np.zeros((height, width, nClasses))
    try:
        labels = cv2.imread(path, 1)
        labels = cv2.resize(labels, (width, height))
        labels = labels[:, : , 0]

        
        for c in range(nClasses):
            seg_labels[: , : , c ] = (labels == c ).astype(int)

    except Exception:
        print(Exception)

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels


def image_labels_generator(images_path, labels_path,  batch_size, n_classes, input_height, input_width, output_height, output_width):
    
    # This function feeds the keras fit_generator function with the dataset(images and annotations)
    
    assert images_path[-1] == '/'
    assert labels_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png")
    images.sort()
    labels = glob.glob(labels_path + "*.jpg") + glob.glob(labels_path + "*.png")
    labels.sort()

    assert len(images) == len(labels)

    z = list(zip(images,labels))
    random.shuffle(z)
    zipped = itertools.cycle(z)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img , label = next(zipped)
            X.append( get_images(img, input_width, input_height)  )
            Y.append(get_labels(label, n_classes, output_width, output_height))

        yield np.array(X) , np.array(Y)
