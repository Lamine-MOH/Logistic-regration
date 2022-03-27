import numpy as np
import json


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
        else:
            self.mixing_formats = ""
            for i in range(1, hypo_level+1):
                self.mixing_formats += self.get_possible_mixing_formats(fet_num, i)

            self.mixing_formats = np.array(self.mixing_formats[:-1].split("\n"))
            self.thetas_num = len(self.mixing_formats) + 1 

        # create the thetas with random values #
        self.thetas = np.random.standard_normal(self.thetas_num)       

    #
    def load_module(self, module_source=""):
        with open(module_source, "r") as f:
            information = json.load(f)

            self.thetas =       information["thetas"]
            self.mixing_formats=np.array(information["mixing_formats"])
            self.fet_num =      int(information["fet_num"])
            self.hypo_level =   int(information["hypo_level"])
            self.mixing =       bool(information["mixing"])
            self.regularized =  bool(information["regularized"])
            self.alpha =        int(information["alpha"])
            self.landau =       int(information["landau"])
            self.thetas_num =   len(self.thetas)
            
            # numerate the thetas #
            for i in range(len(self.thetas)):
                self.thetas[i] = int(self.thetas[i])
            #  #
            self.thetas = np.array(self.thetas)

    #
    def get_possible_mixing_formats(self, fet_num, level, mixing_string=""):
        if(level <= 0):
            return mixing_string + "\n"

        result = ""
        for i in range(fet_num, 0, -1):
            result += self.get_possible_mixing_formats(
                i, level - 1, mixing_string + str(i-1))

        return result

    #
    def predict(self, featers):

        if(self.mixing):
            sum = 0
            
            for format_str, theta in zip(self.mixing_formats, self.thetas[1:]):
                value = 1
                for index in format_str:
                    value *= featers[ int(index) ]
                    
                sum += value * theta
                    
            return sum + self.thetas[0]

        else:
            return np.sum( [(featers**i) * self.thetas[1:]] for i in range(1, self.hypo_level + 1) ) + self.thetas[0]
            
    #
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #
    def J(self, inputs, outputs, X_format="", landau=0):
        m = len(outputs)

        regularization = (landau/2*m) * np.sum( [ theta**2 for theta in self.thetas[1:] ] ) if self.regularized else 0

        result = 0

        for features, y in zip(inputs, outputs):
            value = 1
            for index in X_format:
                value *= features[int(index)]

            predicted_value = self.sigmoid(self.predict(features))
            predicted_value = 0.00000000000001 if predicted_value==0 else predicted_value
            predicted_value = 0.99999999999999 if predicted_value==1 else predicted_value
            
            result += (1/m) * ( y * np.log( predicted_value ) + (1-y)*(np.log( 1-predicted_value )) ) * value + regularization

        return result
    
    def save_progress(self, location="data/learned module.json", iteration=""):
        print("module information Saved " + ("" if iteration=="" else f", iteration: {str(iteration)}"))
        
        thetas_value = []
        for theta in self.thetas:
            thetas_value.append(int(theta))
        
        mixing_formats_value = []
        for formate in self.mixing_formats:
            mixing_formats_value.append(str(formate))
        
        information = {
            "thetas": thetas_value,
            "mixing_formats": mixing_formats_value,
            "fet_num": self.fet_num,
            "hypo_level": self.hypo_level,
            "mixing": self.mixing,
            "regularized": self.regularized,
            "alpha": self.alpha,
            "landau": self.landau
        }

        with open(location, "w") as f:
            json.dump(information, f)

    # 
    def gradient_descent(self, inputs, outputs, iterations_num=1000, saving_rate=200):
        for iteration in range(iterations_num):
            
            self.thetas[0] -= self.alpha * self.J(inputs, outputs)
            for theta_index, format in zip(range(1, self.thetas_num), self.mixing_formats):
                self.thetas[theta_index] -= self.alpha * self.J(inputs, outputs, format, self.landau) 

            if iteration % saving_rate == 0:
                self.save_progress(iteration=iteration)
                
        self.save_progress(iteration=iteration)
                 
    #
    def test(self, inputs, outputs):
        data_length = len(outputs)
        score = 0
        
        for features, y in zip(inputs, outputs):
            guess = self.predict(features)
            guess = 1 if guess >= 0.5 else 0
            
            if y == guess:
                score += 1
                
        return score * 100 / data_length

