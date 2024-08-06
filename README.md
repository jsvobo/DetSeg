# Detection and segmentation for FG-BG image decomposition

This repository is for exploration of different object detection and segmentation approaches.



git clone https://github.com/orrzohar/PROB.git

conda create -n detect
conda activate detect
conda install python 3.11
pip install -r requirements.txt -r ./PROB/requirements.txt

#build conda operators as per https://github.com/orrzohar/PROB

cd ./PROB/models/ops
sh ./make.sh

#DON'T FORGET to switch GPUs !
python test.py