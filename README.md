# U-Lite
PyTorch code for paper [1M parameters are enough? A lightweight CNN-based model for medical image segmentation](https://ieeexplore.ieee.org/document/10317244), APSIPA 2023. 

## Update
* Code is in progress.

## Abtract
Deep learning models often have to deal with a trade-off between the need for high accuracy and the desire for low computational cost. In this work, we look for a lightweight U-Net-based model which can remain the same or even achieve better performance for the medical image segmetation, namely U-Lite.

Main highlights:
* U-Lite ultilizes the criss-cross 7x7 convolutional kernels as the main operator.
* The model contains only 878K parameters, x35 fewer parameters and x6 faster than UNet.

## Citation
```
@INPROCEEDINGS{10317244,
  author={Dinh, Binh-Duong and Nguyen, Thanh-Thu and Tran, Thi-Thao and Pham, Van-Truong},
  booktitle={2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)}, 
  title={1M parameters are enough? A lightweight CNN-based model for medical image segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1279-1284},
  doi={10.1109/APSIPAASC58517.2023.10317244}}
```
