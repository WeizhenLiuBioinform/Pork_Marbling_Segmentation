# Marbling-Net: a novel intelligent framework for pork marbling segmentation using images from smartphones

This repository contains the code and dataset for the paper "Marbling-Net: a novel intelligent framework for pork marbling segmentation using images from smartphones," accepted at MDPI Sensors.

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
# Citation
If you use this code and data in a publication, please cite it as:

* Zhang, S.; Chen, Y.; Liu, W.;Liu, B.; Zhou, X. Marbling-Net: A Novel Intelligent Framework for Pork Marbling
  Segmentation Using Images from Smartphones. Sensors. 2023, 23, 5135. https://doi.org/10.3390/s23115135
