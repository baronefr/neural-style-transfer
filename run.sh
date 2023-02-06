#!/usr/bin/bash

python3 gatys.py -output out.jpg -iterations 300 \
    -content-weight 1e-7 \
    -style image/matisse.jpg \
    -content image/mccurry-afghan.jpg \
    -color-control luminance
    #-optimizer Adam -learn-rate 100 \  # Adam requires higher LR...