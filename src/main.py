import logistic_Regration as lr
import sys
import csv
import numpy as np

####################### import data ####################### 
def get_data(location):
    inputs = []
    outputs = []

    with open(location, "r") as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            for i in range(len(line)):
                line[i] = int(line[i])
            
            inputs.append(line[0 : len(line)-1])
            outputs.append(int(line[len(line)-1]))

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    return inputs, outputs

####################### create model ####################### 

if sys.argv[1] == "new":
# if True:
    
    data_location = sys.argv[2]
    # data_location = "data/learn data.csv"
    inputs, outputs = get_data(data_location)
    
    hypo_level = 3
    alpha = 0.001
    landau = 0.01
    mixing = True
    regularized = False
    iterations_num = 10000
    saving_rate = 200
    
    for arg in sys.argv[3:]:
        if arg.__contains__("hypo_level="):
            hypo_level = int(arg.replace("hypo_level=", ""))

        elif arg.__contains__("alpha="):
            alpha = int(arg.replace("alpha=", ""))

        elif arg.__contains__("landau="):
            landau = int(arg.replace("landau=", ""))

        elif arg.__contains__("iterations_num="):
            iterations_num = int(arg.replace("iterations_num=", ""))
        
        elif arg.__contains__("saving_rate="):
            saving_rate = int(arg.replace("saving_rate=", ""))

        elif arg == "mixing":
            mixing = True
        elif arg == "regularized":
            regularized = True
    
        
    module = lr.Logistic()
    module.new_module(len(inputs[0]), hypo_level, alpha, landau, mixing, regularized)
    module.gradient_descent(inputs, outputs, iterations_num, saving_rate)
    
      
elif sys.argv[1] == "contune":
    
    model = lr.Logistic(thetas_source=sys.argv[2])

    print("thetas: " + str(model.thetas))
    print("mixing_formats: " + str(model.mixing_formats))
    print("fet_num: " + str(model.fet_num))
    print("hypo_level: " + str(model.hypo_level))
    print("mixing: " + str(model.mixing))
    print("regularized: " + str(model.regularized))
    print("alpha: " + str(model.alpha))
    print("landau: " + str(model.landau))
