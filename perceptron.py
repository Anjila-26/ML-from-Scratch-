import numpy as np



class Perceptron : 
    """
    Parameters required : 

    w : weights initialized by the perceptron
    b : bias

    epochs : number of epochs
    random_state : to initialize the random weights
    learning_rate : updating the weights and bias for learning 

    """

    def __init__(self, learning_rate = 0.01, epochs = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self,X,y):

        random = np.random.RandomState(self.random_state)
        self.w = random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.float_(0.)
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for i, target in zip(X,y):
                update = self.learning_rate * (target - self.predict(i)) 
                self.w += update * i
                self.b += update
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def input_calculation(self,X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        return np.where(self.input_calculation(X) >= 0.0, 1, 0)
        
