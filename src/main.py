from pyexpat import features
import logistic_Regration as lr
import sys
import csv
import numpy as np

####################### import data ####################### 
def get_data(location, data_limit=-1):
    if location == "":
        return [], []
    
    inputs = []
    outputs = []

    with open(location, "r") as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            for i in range(len(line)):
                line[i] = float(line[i])
            
            inputs.append(line[0 : len(line)-1])
            outputs.append(float(line[len(line)-1]))
            
            data_limit -= 1
            if data_limit == 0:
                break

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    return inputs, outputs

####################### create model ####################### 

if sys.argv[1] == "new":
    print()
    
    # load the argv variables
    train_data_location = ""
    test_data_location = ""
    data_limit = -1
    hypo_level = 1
    alpha = 0.001
    landau = 0.01
    mixing = False
    regularized = False
    iterations_num = 1000
    saving_rate = 100
    
    for arg in sys.argv[2:]:
        if arg.__contains__("train_data="):
            train_data_location = arg.replace("train_data=", "")
        
        elif arg.__contains__("test_data="):
            test_data_location = arg.replace("test_data=", "")
        
        elif arg.__contains__("data_limit="):
            data_limit = float(arg.replace("data_limit=", ""))

        elif arg.__contains__("hypo_level="):
            hypo_level = int(arg.replace("hypo_level=", ""))

        elif arg.__contains__("alpha="):
            alpha = float(arg.replace("alpha=", ""))

        elif arg.__contains__("landau="):
            landau = float(arg.replace("landau=", ""))

        elif arg.__contains__("iterations_num="):
            iterations_num = int(arg.replace("iterations_num=", ""))
        
        elif arg.__contains__("saving_rate="):
            saving_rate = int(arg.replace("saving_rate=", ""))

        elif arg == "mixing":
            mixing = True
        elif arg == "regularized":
            regularized = True
    
    
    # load the data
    print(f"Loading Train Data from '{train_data_location}' ... ", end="")
    train_inputs, train_outputs = get_data(train_data_location, data_limit)
    print(f"Loading complete , data length: {len(train_outputs)}")

    print(f"Loading Test Data from '{test_data_location}' ... ", end="")
    test_inputs, test_outputs = get_data(test_data_location)
    print(f"Loading complete , data length: {len(test_outputs)}", end="\n\n")
        
    # create the module
    module = lr.Logistic()
    
    print("Creating new module... ", end="")
    module.new_module(len(train_inputs[0]), hypo_level, alpha, landau, mixing, regularized)
    print("Module created", end="\n\n")
    print(f"Number of features: {module.fet_num}")
    print(f"Hypothesis level: {module.hypo_level}")
    print(f"Number of thetas: {module.thetas_num}")
    print("Mixing Formats" if module.mixing else "")
    print("Regularized Active \n" if module.regularized else "")
    
    # start the learning
    print(f"Running gradient descent for {iterations_num} Iterations, with saving avery {saving_rate} Iteration")
    module.gradient_descent(train_inputs, train_outputs, iterations_num, saving_rate, test_inputs = test_inputs, test_outputs = test_outputs)
    print("Learning complete")
    
      
elif sys.argv[1] == "continue":
    
    # load the argv variables
    train_data_location = ""
    test_data_location = ""
    data_limit = -1
    iterations_num = 1000
    saving_rate = 100
    
    for arg in sys.argv[2:]:
        if arg.__contains__("train_data="):
            train_data_location = arg.replace("train_data=", "")
        
        elif arg.__contains__("test_data="):
            test_data_location = arg.replace("test_data=", "")
        
        elif arg.__contains__("data_limit="):
            data_limit = float(arg.replace("data_limit=", ""))

        elif arg.__contains__("iterations_num="):
            iterations_num = int(arg.replace("iterations_num=", ""))
        
        elif arg.__contains__("saving_rate="):
            saving_rate = int(arg.replace("saving_rate=", ""))

    # Load the data
    print(f"Loading Train Data from '{train_data_location}' ... ", end="")
    train_inputs, train_outputs = get_data(train_data_location, data_limit)
    print(f"Loading complete , data length: {len(train_outputs)}")

    print(f"Loading Test Data from '{test_data_location}' ... ", end="")
    test_inputs, test_outputs = get_data(test_data_location)
    print(f"Loading complete , data length: {len(test_outputs)}", end="\n\n")
        
    # Load the module
    module_location = sys.argv[2]
    module = lr.Logistic()
    
    print("Load the module... ", end="")
    module.load_module(module_location)
    print("Module Loaded", end="\n\n")
    print(f"Number of features: {module.fet_num}")
    print(f"Hypothesis level: {module.hypo_level}")
    print(f"Number of thetas: {module.thetas_num}")
    print("Mixing Formats" if module.mixing else "")
    print("Regularized Active \n" if module.regularized else "")
        
    # start the learning
    print(f"Running gradient descent for {iterations_num} Iterations, with saving avery {saving_rate} Iteration")
    module.gradient_descent(train_inputs, train_outputs, iterations_num, saving_rate, test_inputs = test_inputs, test_outputs = test_outputs)
    print("Learning complete")
    
    
elif sys.argv[1] == "test":
    
    # load the argv variables
    test_data_location = ""
    data_limit = -1
    
    for arg in sys.argv[3:]:
        if arg.__contains__("test_data="):
            test_data_location = arg.replace("test_data=", "")
        
        elif arg.__contains__("data_limit="):
            data_limit = float(arg.replace("data_limit=", ""))
    
    # Load the data
    print(f"Loading Test Data from '{test_data_location}' ... ", end="")
    test_inputs, test_outputs = get_data(test_data_location)
    print(f"Loading complete , data length: {len(test_outputs)}", end="\n\n")
        
    
    # Load the module
    module_location = sys.argv[2]
    module = lr.Logistic()
    
    print("Load the module... ", end="")
    module.load_module(module_location)
    print("Module Loaded", end="\n\n")
    print(f"Number of features: {module.fet_num}")
    print(f"Hypothesis level: {module.hypo_level}")
    print(f"Number of thetas: {module.thetas_num}")
    print("Mixing Formats" if module.mixing else "")
    print("Regularized Active \n" if module.regularized else "")
        
    # start the test
    data_length = len(test_outputs)
    
    print(f"Running Test for {data_length} Case... ", end="")
    score = module.test(test_inputs, test_outputs)
    print("Test complete", end="\n\n")
    
    total_correct = score['yes_correct'] + score['no_correct']
    
    print(f"Result: {total_correct*100/data_length}%  , {total_correct}/{data_length}", end="\n\n")
    
    print(f"Number of YES correct: {score['yes_correct']}")
    print(f"Number of NO correct: {score['no_correct']}")
    print(f"Number of YES false: {score['yes_false']}")
    print(f"Number of NO false: {score['no_false']}")

# elif sys.argv[1] == "preview":    
    
#     # Load the module
#     module_location = sys.argv[2]
#     module = lr.Logistic()
    
#     print("Load the module... ", end="")
#     module.load_module(module_location)
#     print("Module Loaded", end="\n\n")
#     print(f"Number of features: {module.fet_num}")
#     print(f"Hypothesis level: {module.hypo_level}")
#     print(f"Number of thetas: {module.thetas_num}")
#     print("Mixing Formats" if module.mixing else "")
#     print("Regularized Active \n" if module.regularized else "")
        
#     features = [
#         28.5,       # bmi       
#         1,          # smoking      
#         0,          # alcohol      
#         1,          # stroke      
#         0,         # physicalHealth      
#         0,         # hentalHealth      
#         1,          # diffWalking      
#         2,          # sex      
#         55,         # age      
#         1,          # race      
#         1,          # diabetic      
#         0,          # physicalActivity      
#         1,          # genHealth      
#         20,         # sleepTime      
#         1,          # asthma      
#         1,          # kidneyDisease      
#         1,          # skinCancer        
#     ] 
      
#     features=  [25.06,  0,    0,    0,    0,    0, 
#    1,    2,   80,    1,    1,    0,
#   3,    7,    0,    0,    1,  ]
    
#     print(module.sigmoid(module.predict(features)))