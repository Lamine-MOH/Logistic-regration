import csv
from random import random
import sys

location = sys.argv[1]
learn_file_size = int(sys.argv[2]) if len(sys.argv) >= 3 and int(sys.argv[2]) >= 0 and int(sys.argv[2]) <= 100 else 70

# calc the data length #
with open(location, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    
    data_length = 0
    for line in csv_reader:
        data_length += 1
#

with open(location, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    with open("learn data.csv", "w", newline="") as csv_learn_file:
        with open("test data.csv", "w", newline="") as csv_test_file:
            csv_learn_writer = csv.writer(csv_learn_file)
            csv_test_writer = csv.writer(csv_test_file)
            
            learn_file_counter = 0
            test_file_counter = 0

            for line in csv_reader:
                # clean data #
                line = line[1:]
                line[0] = 1 if line[0] == "Female" else 2
                #  #
                
                if learn_file_counter >= data_length * learn_file_size / 100:
                    csv_test_writer.writerow(line)
                elif test_file_counter >= data_length * (100 - learn_file_size) / 100:
                    csv_learn_writer.writerow(line)
                else:
                    if random() <= learn_file_size / 100:
                        csv_learn_writer.writerow(line)
                        learn_file_counter += 1
                    else:
                        csv_test_writer.writerow(line)
                        test_file_counter += 1
                        
                        
                    
                    
                
                

