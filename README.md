# gestures_recog_NN

PyTorch implementation of a CNN based gesture recognition system.

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

# Training Set

All the models have been trainend usinge the custom dataset called Briareo.



![alt text](https://github.com/manganarofabio/gestures_recog_NN/blob/master/gesti.jpg)

## Contents
1) main.py provides all the CNN models which have been usedvand training sturucture.
2) dataloeder.py provides the implementation of pytorch dataloaed for a custom dataset.
