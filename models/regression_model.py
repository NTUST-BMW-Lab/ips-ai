from abc import ABC, abstractmethod

class RegressionModel(ABC):
    @abstractmethod
    def train(self, x_train_scaled, train_labels):
        pass

    @abstractmethod
    def predict(self, x_test_scaled, test_labels):
        pass
    
    @abstractmethod
    def evaluate(self, test_labels, predicted_coords):
        pass