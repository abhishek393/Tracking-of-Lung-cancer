# Tracking-of-Lung-cancer

The goal was to create a respiratory monitoring system to detect lung cancer depth. RGB-D camera (Microsoft Kinect for Windows v2) was used for the project. Data was recorded for 18 volunteers(data is not uploaded). The recorded data was in .xef format which was converted to .mat & .jpg files using Xef2Mat-Jpg converter. The converted .mat and .jpg files were saved in folders named in this format ->'sub' + 'serial number'. The .mat files contain the matrices with depth values in millimeters which are converted to depth frames using value%256. Then these depth frames are converted into 224x224 images covering the chest area.

show_colour_frames.py: This file displays all the colour frames for each of the 18 patients/volunteers.

show_depth_frames.py: This file displays all the depth frames for each of the 18 patients/volunteers.

mapping.py: This file is used to map the 512x424 depth frames into 224x224 input for VGGnet which is used in data.py

data.py: This file is used for training and testing. The model used is VGGNet.

labelling.py: This file is used to create labels namely, Below average breathing(0), average breathing(1) and above average breathing(2). it uses a 7x7 cropped matrix from the chest area and checks the changes in the depths to get the labels. Alternatively, you can assign labels on your own. The labels are stored i labels.csv

Tools: Microsoft Kinect for Windows v2 sdk -> https://www.microsoft.com/en-in/download/details.aspx?id=44561

Xef2Mat-Jpg -> https://github.com/LuciaXu/Xef2Mat-Jpg

This repository is meant to demonstrate the flow and not the actual running code. The paths used in the repo are static and not being taken as inputs. Please change the paths accordingly to use this repository.
