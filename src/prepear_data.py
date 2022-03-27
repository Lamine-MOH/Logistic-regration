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

    with open("data/learn data.csv", "w", newline="") as csv_learn_file:
        with open("data/test data.csv", "w", newline="") as csv_test_file:
            csv_learn_writer = csv.writer(csv_learn_file)
            csv_test_writer = csv.writer(csv_test_file)
            
            learn_file_counter = 0
            test_file_counter = 0

            for line in csv_reader:
                
                new_line = []
                new_line.append(line[1])
                new_line.append(1 if line[2] == "Yes" else 0)
                new_line.append(1 if line[3] == "Yes" else 0)
                new_line.append(1 if line[4] == "Yes" else 0)
                new_line.append(line[5])
                new_line.append(line[6])
                new_line.append(1 if line[7] == "Yes" else 0)
                new_line.append(1 if line[8] == "Male" else 2)
                new_line.append(80 if line[9].__contains__("or older") else sum([ int(value) for value in line[9].split("-") ]) / 2)
                new_line.append(1 if line[11].__contains__("Yes") else 0)
                new_line.append(1 if line[12] == "Yes" else 0)
                new_line.append(line[14])
                new_line.append(1 if line[15] == "Yes" else 0)
                new_line.append(1 if line[16] == "Yes" else 0)
                new_line.append(1 if line[17] == "Yes" else 0)
                new_line.append(1 if line[0] == "Yes" else 0)
                
                
                
                if learn_file_counter >= data_length * learn_file_size / 100:
                    csv_test_writer.writerow(new_line)
                elif test_file_counter >= data_length * (100 - learn_file_size) / 100:
                    csv_learn_writer.writerow(new_line)
                else:
                    if random() <= learn_file_size / 100:
                        csv_learn_writer.writerow(new_line)
                        learn_file_counter += 1
                    else:
                        csv_test_writer.writerow(new_line)
                        test_file_counter += 1
                        
                        
                    
                    
                
                

