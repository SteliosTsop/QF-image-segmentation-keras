# Quantitative Fractography Image Segmentation

This repository is created to present the training process and the predictions results that are published on the scientific research work: _" Toward quantitative fractography using convolutional neural networks "_.

The source code is a modification of the code published at [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras), with the addition of some extra tools needed for the training process. 

The main objective of publishing this work is to propose a new method for the topographic charactirization of fracture surfaces based on Convolutional Neural Networks and attract the interest of the Fractography research community in order to built on this basis and develop tools that optimize the Quantitative Fractogrphy techniques. 

More specifically, the Convolutional Neural Network (CNN) model after being trained in Scanning Electron Microscopy (SEM) images of fracture surfaces is able to identify the _intergranular_ or _transgranular_ fracture modes for any brittle material.

<img src="images/SEM_Predictions.jpg">


## Annotation of the training and validation datasets

The first part of the training of every Convolutional Neural Network (CNN) model involveds the annotation of the images. In our case the dataset is composed by SEM images of the fracture surfaces. 

The annotation for the SEM fracture images has been performed with the online open source VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/via.html). Using the polygon tool it becomes possible to label the different areas of the SEM images as _intergranular_ or _transgranular_, while the areas that were more ambiguous or between the borders of adjucent areas were classified as _background_. Furthermore, the image annotation is a very time consuming task and the introduction of the _background_ label was necessary.

<p align="center">
  <img src="images/VGG_annotator.jpg" width="300">
</p>

After annotating around 1000 images (with size 640x640), the next step is to convert the annotations into a format that is suitable for the training program. This is done using the __Export_annotations.py__ script in the __Convert_VGG_Annotations__ folder.  


## Training the network

The code for training the network and performing the predictions is using Keras with Tensorflow as a backend. 
The __train.py__ code is used to train the network and the following command line arguments need to be defined:

- __save_weights_path__ : directory to save the .hdf5 file of the trained weights
- __train_images__ : directory of the training dataset (_/Convert_VGG_Annotations/fracture_images/train_)
- __val_images__ : directory of the validation dataset (_/Convert_VGG_Annotations/fracture_images/val_)
- __train_annotations__ : directory of the annotations of the training dataset (_/Convert_VGG_Annotations/annotations/train_)
- __val_annotations__ : directory of the annotations of the validation dataset (_/Convert_VGG_Annotations/annotations/val_)
- __logs__ : directory of Tensorboard log files
- __n_classes__ : number of classes (including the _background_ class)
- __input_height__ : height in pixels of the train and val images(default value: _640_) 
- __input_width__ : width in pixels of the train and val images(default value: _640_) 
- __start_epoch__ : initial epoch to start the training - _ if it is a new training, use the default value zero_
- __end_epoch__ : final training epoch
- __epoch_steps__ : number of iterations per epoch
- __batch_size__ : depending on the GPU memory of your computer, define the batch size for training
- __val_batch_size__ : depending on the GPU memory of your computer, define the batch size for training
- __init_learning_rate__ : learning rate of training
- __optimizer_name__ : choose optimizer - _options: rmsprop, adadelta, sgd / default: adam    
- __load_weights__ : directory of the pre-trained weights, in case of continuing the training from previous pre-trained stage. No need to define this in case of starting a new training. 

An example of execution command is:

```
python train.py --save_weights_path="weights/trained_weights.hdf5" --logs="logs/" --start_epoch=0 \
                --train_images="Convert_VGG_Annotations/fracture_images/train" \
                --train_annotations="Convert_VGG_Annotations/annotations/train" \
                --val_images="Convert_VGG_Annotations/fracture_images/val" \
                --val_annotations="Convert_VGG_Annotations/annotations/val" \
                --n_classes=3  --optimizer_name="adadelta" \
                --init_learning_rate=0.00008 --input_height=640 --input_width=640
```
