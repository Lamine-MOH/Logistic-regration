import csv
import numpy as np

data_location = "data/learn_data.csv"

data_length = 0
with open(data_location, "r", newline="") as csv_file_read:
    csv_reader = csv.reader(csv_file_read)

    for line in csv_reader:
        data_length += 1

print(f"data_length: {data_length}")


with open(data_location, "r", newline="") as csv_file_read_yes:
    with open(data_location, "r", newline="") as csv_file_read_no:
        csv_reader_yes = csv.reader(csv_file_read_yes)
        csv_reader_no = csv.reader(csv_file_read_no)

        with open("data/test_data_02.csv", "w", newline="") as csv_file_writer:
            csv_writer = csv.writer(csv_file_writer)

            for i in range(int(data_length/2)):

                for yes_row in csv_reader_yes:
                    if int(yes_row[-1]) == 1:
                        csv_writer.writerow(yes_row)
                        break

                for no_row in csv_reader_no:
                    if int(no_row[-1]) == 0:
                        csv_writer.writerow(no_row)
                        break
