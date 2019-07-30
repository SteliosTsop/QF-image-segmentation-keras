# Quantitative Fractography Image Segmentation

This repository is created to present the training process and the predictions results that are published on the scientific research work: _" Toward quantitative fractography using convolutional neural networks "_.

The source code is a modification of the code published at [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras), with the addition of some extra tools needed for the training process. 

The main objective of publishing this work is to propose a new method for the topographic charactirization of fracture surfaces based on Convolutional Neural Networks and attract the interest of the Fractography research community in order to built on this basis and develop tools that optimize the Quantitative Fractogrphy techniques. 

More specifically, the Convolutional Neural Network (CNN) model after being trained in Scanning Electron Microscopy (SEM) images of fracture surfaces is able to identify the _intergranular_ or _transgranular_ fracture modes for any brittle material.

<img src="data/SEM_Predictions.jpg">


## Annotation of the training and validation datasets

The first part of the training of every Convolutional Neural Network (CNN) model involveds the annotation of the images. In our case the dataset is composed by SEM images of the fracture surfaces. 

The annotation for the SEM fracture images has been performed with the online open source VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/via.html). Using the polygon tool it becomes possible to label the different areas of the SEM images as _intergranular_ or _transgranular_, while the areas that were more ambiguous or between the borders of adjucent areas were classified as _background_. Furthermore, the image annotation is a very time consuming task and the introduction of the _background_ label was necessary.

![VIA](data/VGG_annotator.jpg){:height="24px" width="48px"}



