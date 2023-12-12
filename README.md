# Marbling-Net: a novel intelligent framework for pork marbling segmentation using images from smartphones

Our paper is under review.


# The tips for using the Pork IMF App
(1) Clean the subcutaneous and connective tissues in the slices of the LD muscle;
(2) Take the image of pork LD muscle slices in a black background; 
(3) Choose smartphones with high resolution camera; 
（4）Minimize reflections as much as possible when capturing images of LD muscles in pork.

# Deep learning pipeline for Pork Marbling segmentation

## Start
First of all, clone the code
```shell script
git clone https://github.com/WeizhenLiuBioinform/Pork_Marbling_Segmentation.git
```
## Prerequisites
* Python 3.8
* Pytorch 1.6
* CUDA 10.0 or higher
* Opencv
* ...

The complete list of the required python packages and their version information can be found at requirements.txt
## MarblingNet_pytorch

### PMD2023 Data Preparation
* **PASCAL_VOC format**: 
Make sure the ImageSets, JPEGImages and SegmentationClass under the PMD2023 folder.

### config.json settings
Set the parameters required for training in config.json, especially the data set path.

### MarblingNet_Train
To train a MarblingNet model with pascal_voc format, simply run:
```shell script
python train.py
```

### MarblingNet_predict
Use the trained model to make predictions, simply run:
```shell script
python predict.py
```
