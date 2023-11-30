import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        self.hidden_layer_output = None
        self.losses = []


    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2


    # FORWARD PASS
    def forward(self, inputs):
        #finding input to hidden layer here
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.tanh(hidden_layer_input)
        # hidden to output layer or final layer
        output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        final_output = self.tanh(output_layer_input)
        return final_output
    

    # BACKWARD PASS
    def backward(self, inputs, targets, output, learning_rate):
        error = targets - output
        # DELTA ERROR
        output_delta = error * self.tanh_derivative(output)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.tanh_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate


    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            #it triggering forward
            output = self.forward(inputs)
            #it is basically triggering or calling the backward propogation
            self.backward(inputs, targets, output, learning_rate)
            #here we are mean squaring the error and appending the loss
            loss = np.mean(0.5 * (targets - output) ** 2)
            self.losses.append(loss)
        #After which operations we want to see epoch
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, inputs):
        return self.forward(inputs)
    #here we are finding the mean square error, mean absolute error and root mea square error.
    def evaluate(self, predictions, targets):
        MSE = np.mean((targets - predictions) ** 2)
        MAE = np.mean(np.abs(targets - predictions))
        RMSE = np.sqrt(MSE)
        return MSE,MAE,RMSE
    #accuracy finding task is implementing here in this we are taking data in the form of binary data
    def accuracy(self, predictions, targets, threshold=0.5):
        binary_predictions = (predictions > threshold).astype(int)
        accuracy = np.mean(binary_predictions == targets.reshape(-1, 1))
        return accuracy

# HEART DISEASE DATASET
heart_data = pd.read_csv('/Users/gamingspectrum24/Documents/University Coursework/5th Semester/Neural Networks Lab/NN from Scratch/Heart_disease.csv')

# PREPROCESSING
threshold_bloodpressure = 110
heart_data['target'] = (heart_data['trestbps'] > threshold_bloodpressure).astype(int)


features = heart_data[['age', 'cp', 'chol']]
target = heart_data['target']

scaler = StandardScaler() # SCIKIT LEARN
features_scaled = scaler.fit_transform(features)  #transform is for computation mean and standard deviation in dataonly

#here data is splitting in train and test
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42) 

#self defined parameter used in NN
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.01

#Calling NeuralNetwork
nn = NeuralNetwork(input_size, hidden_size, output_size)

#calling training model
nn.train(X_train, y_train.values.reshape(-1, 1), epochs=50, learning_rate=learning_rate)

#calling Predictions testing
predictions = nn.predict(X_test)

#calling evaluation function like mean square error, mean absolute error and root mean square error and Accuracy part
MSE, MAE, RMSE = nn.evaluate(predictions, y_test.values.reshape(-1, 1))
accuracy = nn.accuracy(predictions, y_test.values.reshape(-1, 1))

#Output we are showing
print(f"Mean Squared Error: {MSE:.4f}")
print(f"Mean Absolute Error: {MAE:.4f}")
print(f"Root Mean Squared Error: {RMSE:.4f}")
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")   