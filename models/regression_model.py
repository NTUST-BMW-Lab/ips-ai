from abc import ABC, abstractmethod

class RegressionModel(ABC):
    """
    An abstract base class representing a regression model for predicting coordinates.

    Methods:
        train(x_train_scaled, train_labels):
            Abstract method to train the regression model using the given training data and labels.

        predict(x_test_scaled, test_labels):
            Abstract method to predict the coordinates for the given test data.

        evaluate(test_labels, predicted_coords):
            Abstract method to evaluate the model's performance using Mean Squared Error (MSE) 
            and R-squared (R2) metrics and return the results.

    Note:
        - This is an abstract class and should not be instantiated directly.
        - Subclasses must implement the abstract methods 'train', 'predict', and 'evaluate'.
        - 'x_train_scaled' and 'x_test_scaled' are the scaled feature matrices for training and testing, respectively.
        - 'train_labels' and 'test_labels' are the labels/coordinates corresponding to the training and testing data.
        - The 'train' method trains the regression model using the given training data and labels.
        - The 'predict' method uses the trained model to predict the coordinates for the given test data.
        - The 'evaluate' method calculates and returns the model's performance using MSE and R2 metrics.

    Example Usage:
        class MyCustomModel(RegressionModel):
            def train(self, x_train_scaled, train_labels):
                # Implement your custom training logic here

            def predict(self, x_test_scaled, test_labels):
                # Implement your custom prediction logic here
                return predicted_coords

            def evaluate(self, test_labels, predicted_coords):
                # Implement your custom evaluation logic here
                return mse_val, r2_val

        # Instantiate your custom model
        my_model = MyCustomModel()
        # Train the model
        my_model.train(x_train_scaled, train_labels)
        # Predict using the model
        predictions = my_model.predict(x_test_scaled, test_labels)
        # Evaluate the model's performance
        mse_val, r2_val = my_model.evaluate(test_labels, predictions)
    """
    
    @abstractmethod
    def train(self, x_train_scaled, train_labels):
        pass

    @abstractmethod
    def predict(self, x_test_scaled, test_labels):
        pass
    
    @abstractmethod
    def evaluate(self, test_labels, predicted_coords):
        pass