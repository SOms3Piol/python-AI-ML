import numpy as np

class Neuron:
    def __init__(self):
        self.w = np.random.rand(2)  
        self.b = np.random.rand(1)  
        self.output = None

    def sigmoid(self, x):
       
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
       
        self.z = np.dot(X, self.w) + self.b
        self.a = self.sigmoid(self.z)
        self.output = self.a
        return self.output

    def backward(self, X, y, output, learning_rate=1.0):
      
        self.error = y - output
        self.delta = self.error * self.sigmoid_derivative(output)
        self.w += learning_rate * np.dot(X.T, self.delta)
        self.b += learning_rate * np.sum(self.delta, axis=0)

    def train(self, X, y, epochs=1000, learning_rate=1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #y = [0, 1, 1, 1]   OR GATE
    #y = [0 , 0 , 0 , 1] AND GATE

    neuron = Neuron()
    neuron.train(X, y)
    
    print("\nTesting:")
    for i in range(len(X)):
        pred = neuron.forward(X[i])
        print(f"Input: {X[i]}, Prediction: {round(pred[0])}, Target: {y[i]}")
