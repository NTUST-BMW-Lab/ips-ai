from abc import ABC, abstractmethod

class RegressionModel(ABC):
    @abstractmethod
    def train(self, x_train_scaled, train_labels):
        pass

    @abstractmethod
    def predict(self, x_test_scaled, test_labels):
        pass
    
    @abstractmethod
    def evaluate(self, testing_data, predicted_coords):
        pass

    @abstractmethod
    def save(self, model_name, mse_val, r2_val, predicted_coords, folder_dest='../evaluation'):
        pass