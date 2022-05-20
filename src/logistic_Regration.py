import numpy as np
import json
import os.path

class Logistic:
    def new_module(self, fet_num, hypo_level=1, alpha=0.001, landau=0.2, mixing=False, regularized=False):
        self.fet_num = fet_num
        self.hypo_level = hypo_level
        self.mixing = mixing
        self.regularized = regularized
        self.alpha = alpha
        self.landau = landau

        if(not mixing):
            self.thetas_num = fet_num * hypo_level + 1
            
            self.mixing_formats = []
            for i in range(hypo_level):
                for index in range(fet_num):
                    self.mixing_formats.append( [str(index) for j in range(i+1)] )
            self.mixing_formats = np.array(self.mixing_formats, dtype="object")        
            
        else:
            self.mixing_formats = ""
            for i in range(1, hypo_level+1):
                self.mixing_formats += self.get_possible_mixing_formats(fet_num, i)

            self.mixing_formats = np.array( [ format[:-1].split(",") for format in self.mixing_formats[:-1].split("\n") ] , dtype="object")
            self.thetas_num = len(self.mixing_formats) + 1 

        # create the thetas with random values #
        self.thetas = np.random.standard_normal(self.thetas_num)       

    #
    def load_module(self, module_source=""):
        with open(module_source, "r") as f:
            information = json.load(f)

            self.thetas =       information["thetas"]
            self.mixing_formats=np.array(information["mixing_formats"], dtype=object)
            self.fet_num =      int(information["fet_num"])
            self.hypo_level =   int(information["hypo_level"])
            self.mixing =       bool(information["mixing"])
            self.regularized =  bool(information["regularized"])
            self.alpha =        float(information["alpha"])
            self.landau =       float(information["landau"])
            self.thetas_num =   len(self.thetas)
            
            # numerate the thetas #
            for i in range(len(self.thetas)):
                self.thetas[i] = float(self.thetas[i])
            #  #
            self.thetas = np.array(self.thetas)

    #
    def get_possible_mixing_formats(self, fet_num, level, mixing_string=""):
        if(level <= 0):
            return mixing_string + "\n"

        result = ""
        for i in range(fet_num, 0, -1):
            result += self.get_possible_mixing_formats(
                i, level - 1, mixing_string + str(i-1) + ",")

        return result

    #
    def predict(self, features):

        sum = 0
        
        for format_str, theta in zip(self.mixing_formats, self.thetas[1:]):
            value = 1
            for index in format_str:
                value *= features[ int(index) ]
                
            sum += value * theta
                
        return sum + self.thetas[0]

            
    #
    def sigmoid(self, x):
        x = 700 if x>700 else -700 if x<-700 else x
        return 1 / (1 + np.exp(-x))

    #
    def J(self, inputs, outputs):
        m = len(outputs)

        regularization = (self.landau/2*m) * np.sum( [ theta**2 for theta in self.thetas[1:] ] ) if self.regularized else 0

        result = 0

        for features, y in zip(inputs, outputs):

            predicted_value = self.sigmoid(self.predict(features))
            predicted_value = 0.00000000000001 if predicted_value==0 else predicted_value
            predicted_value = 0.99999999999999 if predicted_value==1 else predicted_value
            
            result += ( y * np.log( predicted_value ) + (1-y)*(np.log( 1-predicted_value )) ) + regularization

        return -(1/m) * result
    
    #
    def updating_theta_value(self, inputs, outputs, X_format=""):
        m = len(outputs)

        regularization = (self.landau/2*m) * np.sum( [ theta**2 for theta in self.thetas[1:] ] ) if self.regularized else 0

        result = 0

        for features, y in zip(inputs, outputs):
            value = 1
            for index in X_format:
                value *= features[int(index)]
            
            result += (1/m) * ( self.sigmoid(self.predict(features)) - y ) * value + regularization

        return result
    
    def save_progress(self, iteration="", J="", score = 0):
        # print the iteration state
        print("module information Saved " + ("" if iteration=="" else f", iteration: {str(iteration)}") + ",  J value: " + str(J))
        print()
        
        if score != 0:
            data_length = score['yes_correct']+ score['no_correct']+ score['yes_false']+ score['no_false']
            total_correct = score['yes_correct'] + score['no_correct']
            print(f"Score: {total_correct*100/data_length}%  , {total_correct}/{data_length}")
            print(f"Number of YES correct: {score['yes_correct']}")
            print(f"Number of NO correct: {score['no_correct']}")
            print(f"Number of YES false: {score['yes_false']}")
            print(f"Number of NO false: {score['no_false']}")
            print()
            
        print("---------------------------------------------------------------")
        
        # save the module
        thetas_value = []
        for theta in self.thetas:
            thetas_value.append(float(theta))
        
        mixing_formats_value = []
        for formate in self.mixing_formats:
            mixing_formats_value.append([ index for index in formate])
        
        information = {
            "thetas": thetas_value,
            "mixing_formats": mixing_formats_value,
            "fet_num": self.fet_num,
            "hypo_level": self.hypo_level,
            "mixing": self.mixing,
            "regularized": self.regularized,
            "alpha": self.alpha,
            "landau": self.landau,
            "J": J
        }

        if score == 0:
            with open("result/module.json", "w") as f:
                json.dump(information, f)
        else:
            with open(f"result/module ({iteration}).json", "w") as f:
                json.dump(information, f)
            
        # save the module history
        if os.path.isfile("result/module_history.json"):
            json_file = open("result/module_history.json", "r", encoding="utf-8")
            json_data = json.load(json_file)
            json_file.close()
        else:
            json_data = {"module_history": []}

        json_data["module_history"].append( {
                "iteration": iteration,
                "J_value": J,
                "score": f"{total_correct*100/data_length}%",
                "Number_of_YES_correct" : score['yes_correct'],
                "Number_of_NO_correct" : score['no_correct'],
                "Number_of_YES_false" : score['yes_false'],
                "Number_of_NO_false" : score['no_false'],
            } )
        
        json_file = open("result/module_history.json", "w", encoding="utf-8")
        json.dump(json_data, json_file)
        json_file.close()
            
        
    # 
    def gradient_descent(self, inputs, outputs, iterations_num=1000, saving_rate=200, test_inputs = [], test_outputs = []):
        for iteration in range(iterations_num):
            
            temp = []
            
            temp.append(self.updating_theta_value(inputs, outputs))
            
            for format in self.mixing_formats:
                temp.append(self.updating_theta_value(inputs, outputs, format)) 

            # update the thetas
            for i in range(len(temp)):
                self.thetas[i] -= self.alpha * temp[i]
                
            # save the module
            if iteration % saving_rate == 0:
                self.save_progress(iteration=iteration, J=self.J(inputs, outputs), score = self.test(test_inputs, test_outputs))
                
        self.save_progress(iteration=iteration, J=self.J(inputs, outputs), score = self.test(test_inputs, test_outputs))
                 
    #
    def test(self, inputs, outputs):
        
        yes_correct = 0
        yes_false = 0
        no_correct = 0
        no_false = 0
        
        for features, y in zip(inputs, outputs):
            guess = self.sigmoid(self.predict(features))
            guess = 1 if guess >= 0.5 else 0
            
            if y == 1:
                if guess == 1:
                    yes_correct += 1
                else:
                    yes_false += 1
            else:
                if guess == 0:
                    no_correct += 1
                else:
                    no_false += 1
                
        return { "yes_correct": yes_correct, "yes_false": yes_false, "no_correct": no_correct, "no_false": no_false }

