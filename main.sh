#!/bin/bash
python datasets/prepare.py --root "/media/dustin/DATA/Research/Video-based ReID" --dataset_name vccr
python train.py
python test.py