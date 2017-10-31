#!/usr/bin/env python

import numpy as np
import os
import random
from pdb import set_trace as brake
import math
from PIL import Image


images = os.listdir('/home/nmd89/Downloads/img_align_celeba')

for i in range(len(images)):

    img = Image.open('/home/nmd89/Downloads/img_align_celeba/'+images[i])
    # make the images 128x128
    img = img.crop((89-64, 109-64, 89+64, 109+64))
    img.save('/home/nmd89/Pictures/celeb_images/'+images[i])
    img.close()


