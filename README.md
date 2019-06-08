# gestures_recog_NN

PyTorch implementation of a CNN based gesture recognition system.

This repo is about the project I've done and presented for my Masters' degree thesis. 
All the details and techincal analysis are explained in my [thesis](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/Tesi_Manganaro_Fabio.pdf) and [presentation](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/Manganaro_Fabio_tesi_magistrale_esposizione.pdf)

The whole thesis project leaded to the paper [Hand Gestures for the Human-Car Interaction:the Briareo dataset](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/ICIAP19___Hand_Gestures.pdf) submitted to ICIAP 2019.

Example of working prototype:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YOUTUBE_VIDEO_ID_HERE
" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

## Implemented Models
The gesture recogntion system exploits 4 different neural network architectures implemented in models.py and cross_mode_net.py

1) CNN 2D
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/densenet2.jpg)
2) Multi modal net
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/cross_depth_ir_rgb.png)
3) C3D
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/C3D_model.png)
4) LSTM
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/imgs/lstm_model.png)

## Training Set

All the models have been trainend usinge the custom dataset called Briareo (read [Hand Gestures for the Human-Car Interaction:the Briareo dataset](https://github.com/manganarofabio/gestures_recog_NN/blob/master/docs/ICIAP19___Hand_Gestures.pdf)) which has been collected using the following implemented framework: .


Example of dataset images
![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/gesti.jpg)

## Contents
1) main.py provides all the CNN models which have been used and training sturucture.
2) dataloeder.py provides the implementation of pytorch dataloaed for a custom dataset.
