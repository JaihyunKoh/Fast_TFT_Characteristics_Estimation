# Compensating Non-uniform OLED Pixel Brightness in a Vertical Blanking Interval by Learning TFT Characteristics

## Overview
Project Page : http://data.snu.ac.kr/OLED_external_compensation/ <br>
Paper : https://ieeexplore.ieee.org/abstract/document/9437473

## Requirements
Python >= 3.0 <br>
Pytorch >= 1.1.0

## Usage
1. Download data from [here](https://drive.google.com/file/d/17Pr7rX17_iRrGtgGMpMid5WeGJwIyVhS/view?usp=sharing) and put them in ```/data```.
2. Define ```image_path``` and ```output_file_names``` in ```compensation_and_simulation.py```
3. Run ```compensation_and_simulation.py```

## Folder description
```/model``` contains a RCTCN model written by Pytorch.  
```/checkpoint``` contains a pretrained weight set for RCTCN model.

## Citation
```
@article{koh2021compensating,<br>
    title={Compensating Nonuniform OLED Pixel Brightness in a Vertical Blanking Interval by Learning TFT Characteristics},<br>
     author={Koh, Jaihyun and Kang, Kyeongsoo and Shin, Chaehun and Lee, Soo-Yeon and Yoon, Sungroh},<br>
     journal={IEEE Transactions on Electron Devices},<br>
    volume={68},<br>
    number={7},<br>
    pages={3396--3402},<br>
     year={2021},<br>
     publisher={IEEE}<br>
}
```
