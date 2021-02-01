# Compensating Non-uniform OLED Pixel Brightness in a Vertical Blanking Interval by Learning TFT Characteristics

## Requirements
Python >= 3.0,  
Pytorch >= 1.1.0

## Usage
1. Download data from https://drive.google.com/file/d/17Pr7rX17_iRrGtgGMpMid5WeGJwIyVhS/view?usp=sharing and put them in /data folder.
2. Define "image_path" and "output_file_names" in compensation_and_simulation.py
3. Run compensation_and_simulation.py

## Folder description
/model contains a RCTCN model written by Pytorch.  
/checkpoint contains a pretrained weight set for RCTCN model.
