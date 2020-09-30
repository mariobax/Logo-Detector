
import os
import random

DATA_DIR = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/joined_data/"

BRANDS_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/brands.txt"

TRAIN_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/train.txt"
VAL_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/val.txt"
TEST_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/test.txt"

TRAIN_LOGO_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/train_logo.txt"
VAL_LOGO_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/val_logo.txt"
TEST_LOGO_FILENAME = "/home/mario/Documents/Stanford/ML_Projects/Logo-Detector/data/test_logo.txt"

VAL_SIZE = 1000
TEST_SIZE = 1000


def parse_xml(xml_file):
    logo_locs = []
    with open(xml_file, "r") as f:
        for line in f:
            if "<name>" in line:
                logo_loc = []
                logo = line.split(">")[1].split("<")[0]
                logo_loc.append(logo)
            if "<xmin>" in line:
                x_min = int(line.split(">")[1].split("<")[0])
                logo_loc.append(x_min)
            if "<xmax>" in line:
                x_max = int(line.split(">")[1].split("<")[0])
                logo_loc.append(x_max)
            if "<ymin>" in line:
                y_min = int(line.split(">")[1].split("<")[0])
                logo_loc.append(y_min)
            if "<ymax>" in line:
                y_max = int(line.split(">")[1].split("<")[0])
                logo_loc.append(y_max)
                logo_locs.append(logo_loc)
    return logo_locs
            
                 
        
def create_brands_file():
    brand_names = []
    for file in os.listdir(DATA_DIR):
        if os.path.splitext(file)[1] == ".xml":
            parsed_xml = parse_xml(os.path.join(DATA_DIR, file))
            for brand_loc in parsed_xml:
                brand_name = brand_loc[0]
                if brand_name not in brand_names:
                     brand_names.append(brand_name)
            
    with open(BRANDS_FILENAME, "w") as brands_file:
        for brand in brand_names:
            brands_file.write(brand + "\n")
            
            
def write_split(out_file, data, is_labelled):
    with open(out_file, "w") as file:
        for elem in data:
            line = elem[0] + " "
            for loc_label in elem[1]:
                if not is_labelled:
                    loc_label[-1] = 0
                line += ",".join([str(x) for x in loc_label]) + " "
            line += "\n"
            file.write(line)
            print(line)
            
            
            
def create_train_val_test_splits():
    
    # Load brands into a dict
    brands = {}
    with open(BRANDS_FILENAME, 'r') as brands_file:
        for i, line in enumerate(brands_file):
            brand = line[:-1]
            brands[brand] = i
            
    
    all_data = [] # List of [filename, [[xmin, xmax, ymin, ymax, label]]]
    for file in os.listdir(DATA_DIR):
        if os.path.splitext(file)[1] == ".xml":
            parsed_xml = parse_xml(os.path.join(DATA_DIR, file))
            data = [os.path.splitext(os.path.join(DATA_DIR, file))[0] + ".jpg"]
            brand_loc_data = []
            for brand_loc in parsed_xml:
                xmin = brand_loc[1]
                xmax = brand_loc[3]
                ymin = brand_loc[2]
                ymax = brand_loc[4]
                brand = brands[brand_loc[0]]
                brand_loc_data.append([xmin, ymin, xmax, ymax, brand])
            data.append(brand_loc_data)
            all_data.append(data)
            
    random.shuffle(all_data)
    
    test_set = all_data[: TEST_SIZE]
    val_set = all_data[TEST_SIZE : TEST_SIZE + VAL_SIZE]
    train_set = all_data[TEST_SIZE + VAL_SIZE :]
    
    write_split(TEST_FILENAME, test_set, False)
    write_split(VAL_FILENAME, val_set, False)
    write_split(TRAIN_FILENAME, train_set, False)
    
    write_split(TEST_LOGO_FILENAME, test_set, True)
    write_split(VAL_LOGO_FILENAME, val_set, True)
    write_split(TRAIN_LOGO_FILENAME, train_set, True)
    
    
    
              
        
if __name__ == '__main__':
    
    create_brands_file()
    
    create_train_val_test_splits()