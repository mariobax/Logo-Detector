#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:20:10 2020

@author: mario
"""

import os
from create_data_files import parse_xml
from collections import Counter
from PIL import Image 

CROPPED_LOGO_DIR = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/logo_crops/"
DATA_DIR = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/joined_data/"

def save_image_crop(image_in, image_out, crop_location):
  
    og_im = Image.open(image_in).convert('RGB')
    crop_im = og_im.crop(crop_location) 
    crop_im.save(image_out)
    

if __name__ == "__main__":
    
    # Make sure the directory is created
    if not os.path.exists(CROPPED_LOGO_DIR):
        os.makedirs(CROPPED_LOGO_DIR)
        
    brand_counter = Counter()
        
    data_files = os.listdir(DATA_DIR)
    for data_file in data_files:
        file_name = os.path.splitext(data_file)[0]
        if os.path.splitext(data_file)[1] == ".xml":
            parsed_xml = parse_xml(os.path.join(DATA_DIR, data_file))
            for brand_loc in parsed_xml:
                brand_name = brand_loc[0]
                loc = brand_loc[1:]
                # Make sure the logo crop dir exists
                brand_crops_dir = os.path.join(CROPPED_LOGO_DIR, brand_name)
                if not os.path.exists(brand_crops_dir):
                    os.makedirs(brand_crops_dir)
                    
                crop_save_name = os.path.join(CROPPED_LOGO_DIR, brand_name, file_name + ".jpg")
                
                save_image_crop(os.path.join(DATA_DIR, file_name + ".jpg"), crop_save_name, loc)
                
                brand_counter[brand_name] += 1