import csv
from random import random
import sys

data_location = sys.argv[1]
training_file_size = int(sys.argv[2]) if len(sys.argv) >= 3 and int(sys.argv[2]) >= 0 and int(sys.argv[2]) <= 100 else 75

# calc the data length #
yes_counter = 0
no_counter = 0
with open(data_location, "r") as csv_file: # open the data file for reading
    csv_reader = csv.reader(csv_file) # create csv reader
    next(csv_reader) # skip the features names

    for line in csv_reader:
        if line[0] == "Yes":
            yes_counter += 1
        else:
            no_counter += 1
            
data_length = yes_counter*2 if yes_counter < no_counter else no_counter*2
#

csv_file_read_for_yes_values = open(data_location, "r", newline="") # create a file reader for the yes values
csv_file_read_for_no_values = open(data_location, "r", newline="")  # create a file reader for the no values

csv_yes_reader = csv.reader(csv_file_read_for_yes_values) # create a csv reader for the yes values
csv_no_reader = csv.reader(csv_file_read_for_no_values)   # create a csv reader for the yes values
next(csv_yes_reader) # skip the featers names
next(csv_no_reader)  # skip the featers names
#

csv_training_file = open("data/training_data.csv", "w", newline="") # create a file writer for training data
csv_test_file  = open("data/test_data.csv", "w", newline="")        # create a file writer for test data

csv_training_writer = csv.writer(csv_training_file) # create a csv writer for trainign data
csv_test_writer = csv.writer(csv_test_file)         # create a csv writer for test data
#

def convert_line(line):
    new_line = []
    
    for feature in line[1:]:
        # for the Smoking, AlcoholDrinking, Stroke, DiffWalking, Diabetic, PhysicalActivity, Asthma, KidneyDisease, SkinCancer columns
        if feature.__contains__("Yes"):
            new_line.append(1)
        elif feature.__contains__("No"):
            new_line.append(0)
        
        # for the six column
        elif feature == "Male":
            new_line.append(1)
        elif feature == "Female":
            new_line.append(2)
        #
            
        # for the AgeCategory column
        elif feature.__contains__("-"):
            new_line.append( sum([ int(value) for value in line[9].split("-") ]) / 2 ) # the Average
        elif feature.__contains__(" or older"):
            new_line.append( feature.replace(" or older", "") )
        #
        
        # for the Race column
        elif feature.__contains__("White"):
            new_line.append(1)
        elif feature.__contains__("Black"):
            new_line.append(2)
        elif feature.__contains__("American Indian/Alaskan Native"):
            new_line.append(3)
        elif feature.__contains__("Asian"):
            new_line.append(4)
        elif feature.__contains__("Hispanic"):
            new_line.append(5)
        elif feature.__contains__("Other"):
            new_line.append(6)
        #
        
        # for the GenHealth column
        elif feature.__contains__("Poor"):
            new_line.append(1)
        elif feature.__contains__("Fair"):
            new_line.append(2)
        elif feature.__contains__("Good"):
            new_line.append(3)
        elif feature.__contains__("Very good"):
            new_line.append(4)
        elif feature.__contains__("Excellent"):
            new_line.append(5)
            
        # for the BMI, PhysicalHealth, MentalHealth, SleepTime columns
        else:
            new_line.append(feature)
    
    # add the output at the end (HeartDisease)
    if line[0].__contains__("Yes"):
        new_line.append(1)
    elif line[0].__contains__("No"):
        new_line.append(0)
    #
    
    return new_line

def write(writer):
    for line in csv_yes_reader:
        if line[0] == "Yes":
            writer.writerow( convert_line(line) ) # write line with yes output
            break
            
    for line in csv_no_reader:
        if line[0] == "No":
            writer.writerow( convert_line(line) )# write line with no output
            break
# 

training_counter = int(data_length * (training_file_size/100))
test_counter = data_length - training_counter

for i in range(int(data_length/2)):            
    
    # split the data randomly
    if training_counter <= 0:
        write(csv_test_writer)
        test_counter -= 2
    elif test_counter <= 0:
        write(csv_training_writer)
        training_counter -= 2
    else:
        if random() <= training_file_size / 100:
            write(csv_training_writer)
            training_counter -= 2
        else:
            write(csv_test_writer)
            test_counter -= 2
            
csv_file_read_for_yes_values.close()
csv_file_read_for_no_values.close()
csv_training_file.close()
csv_test_file.close()