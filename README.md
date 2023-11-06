# gestures_recog_NN

PyTorch implementation of a CNN based gesture recognition system.

This repo is about the project I've done and presented for my Master's degree thesis. 
All the details and techincal analysis are explained in my [thesis](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/Tesi_Manganaro_Fabio.pdf) and [presentation](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/Manganaro_Fabio_tesi_magistrale_esposizione.pdf).

The whole thesis project leaded to the submission of the paper ["Hand Gestures for the Human-Car Interaction:the Briareo dataset"](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/ICIAP19___Hand_Gestures.pdf) Fabio Manganaro, Stefano Pini, Guido Borghi, Roberto Vezzani, and Rita Cucchiara. University of Modena and Reggio Emilia, Italy, Department of Engineering “Enzo Ferrari", submitted to ICIAP 2019.

### Important
For more info about the project and instructions to download the dataset please refer to http://imagelab.ing.unimore.it/briareo.

Video example of working prototype:
[![video example](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/th0.png)](https://vimeo.com/341095270)
## Implemented Models
The gesture recogntion system exploits 4 different neural network architectures implemented in models.py and cross_mode_net.py

1. CNN 2D

<img src="https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/densenet2.jpg">

2. Multi modal net

<img src="https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/cross_depth_ir_rgb.png" width="500" height="300">

3. C3D

<img src="https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/C3D_model.png">

4. LSTM
<img src="https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/lstm_model.png" width="500" height="300" >

## Training Set

All the models have been trainend usinge the custom dataset called Briareo (read [Hand Gestures for the Human-Car Interaction:the Briareo dataset](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/ICIAP19___Hand_Gestures.pdf)) which has been collected using the following implemented framework: [gestures collector](https://github.com/manganarofabio/gestures-collector).


Example of gestures in the dataset.
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/gesti.jpg)

## System prototype

The system prototype has been implementented in the project: [demo_gesture_recog](https://github.com/manganarofabio/demo_gesture_recog).

## Contents
1) main.py provides all the CNN models which have been used and training sturucture.
2) dataloeder.py provides the implementation of pytorch dataloaed for a custom dataset.
