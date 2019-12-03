import numpy as np
import pandas as pd


class NeuralNetwork(object):
    """
    Attributes:
        weights_input_to_hidden (np.array):  [n_features, n_hidden_nodes]
        weights_hidden_to_output (np.array):  [n_hidden_nodes, n_output_nodes]
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.n_features = input_nodes
        self.n_hidden_nodes = hidden_nodes
        self.n_output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.n_features ** -0.5,
                                                        (self.n_features, self.n_hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.n_hidden_nodes ** -0.5,
                                                         (self.n_hidden_nodes, self.n_output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """ Train the network on batch of features and targets.

        Args:
            features (np.array): [n_records, n_features]
            targets(np.array): [n_records, n_output_nodes]
        """

        # Ensure features is a numpy array with dimensions [n_records, n_features]
        features = self.check_features(features)
        n_records = features.shape[0]

        # Ensure targets is a numpy array with dimensions [n_records, n_output_nodes]
        if isinstance(targets, (pd.DataFrame, pd.Series)):
            targets = targets.values
        elif isinstance(targets, list):
            targets = np.array(targets)
        elif isinstance(targets, (float, int)):
            targets = np.array([targets])

        if len(targets.shape) == 1:
            targets = targets.reshape(n_records, self.n_output_nodes)
        elif n_records != self.n_output_nodes:
            if targets.shape[0] == self.n_output_nodes:
                targets = targets.reshape(n_records, self.n_output_nodes)
            elif targets.shape[1] != self.n_output_nodes:
                print("target dimensions don't match number of output nodes.")
                raise RuntimeError

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for x, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(x)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, x, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, x):
        """ Implement forward pass here

        Note:
            weights_input_to_hidden:   [n_features, n_hidden_nodes]
            weights_hidden_to_output:  [n_hidden_nodes, n_output_nodes]

            hidden_inputs:  [1, n_hidden_nodes]
            hidden_outputs: [1, n_hidden_nodes]

            final_inputs:  [1, n_output_nodes]
            final_outputs: [1, n_output_nodes]

        Args:
            x (np.array): [n_features] features for a single record

        Returns:
            np.array: [1, n_output_nodes] final outputs
            np.array: [1, n_hidden_nodes] hidden outputs
        """
        hidden_inputs = x.reshape(1, self.n_features).dot(self.weights_input_to_hidden.reshape(self.n_features,
                                                                                               self.n_hidden_nodes))
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output.reshape(self.n_hidden_nodes,
                                                                                self.n_output_nodes))
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_output, hidden_output, x, y, delta_weights_i_h, delta_weights_h_o):
        """ Implement backpropagation

        Notes:
            output_error_term:         [1, n_output_nodes]
            weights_hidden_to_output:  [n_hidden_nodes, n_output_nodes]
            hidden_error:              [1, n_hidden_nodes]
            hidden_error_term:         [1, n_hidden_nodes]

        Args:
            final_output (np.array):  [1, n_output_nodes] output from forward pass
            hidden_output (np.array): [1, n_hidden_nodes] output from forward pass
            x (np.array): [n_features] features for a single record
            y (np.array): [n_output_nodes] targets for a single record
            delta_weights_i_h (np.array): [n_features, n_hidden_nodes] change in weights from input to hidden layers
            delta_weights_h_o (np.array): [n_hidden_nodes, n_output_nodes] change in weights from hidden to output

        Returns:
            np.array: [n_features, n_hidden_nodes] Updated delta_weights_i_h
            np.array: [n_hidden_nodes, n_output_nodes] Updated delta_weights_h_o
        """

        # Enforce shape
        final_output = final_output.reshape(1, self.n_output_nodes)
        hidden_output = hidden_output.reshape(1, self.n_hidden_nodes)

        error = y.reshape(1, self.n_output_nodes) - final_output

        # f(x) = x for final activation function so derivative is 1, unlike sigmoid
        # output_error_term = error * final_output * (1 - final_output)
        output_error_term = error

        # f(x) = x for final activation function so derivative is 1, unlike sigmoid
        # hidden_error = self.weights_hidden_to_output.T * output_error_term * hidden_output * (1 - hidden_output)
        hidden_error = output_error_term.dot(self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        delta_weights_h_o += np.dot(hidden_output.T, output_error_term)
        delta_weights_i_h += x.reshape(self.n_features, 1).dot(hidden_error_term)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step

        Args:
            delta_weights_i_h (np.array): [n_features, n_hidden_nodes] change in weights from input to hidden layers
            delta_weights_h_o (np.array): [n_hidden_nodes, n_output_nodes] change in weights from hidden to output
            n_records (int): number of records
        """
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        self.lr = max(0.999 * self.lr, 0.6)

    def check_features(self, features):
        """ Ensures the features are a numpy array with correct dimensions

        Args:
            features (np.array): [n_records, n_features]

        Returns:
            np.array:  [n_records, n_features] Updated features array
        """

        if isinstance(features, (pd.DataFrame, pd.Series)):
            features = features.values
        elif isinstance(features, list):
            features = np.array(features)
        elif isinstance(features, (float, int)):
            features = np.array([features])

        if len(features.shape) == 1:
            features = features.reshape(1, self.n_features)
        elif features.shape[0] == self.n_features:
            features = features.reshape(features.shape[1], self.n_features)
        elif features.shape[1] != self.n_features:
            print("feature dimensions don't match number of features.")
            raise RuntimeError

        return features

    def run(self, features):
        """ Run a forward pass through the network with input features

        Args:
            features (np.array): [n_records, n_features]

        Returns:
            np.array:  [n_records, n_output]
        """

        features = self.check_features(features)

        hidden_inputs = features.dot(self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = hidden_outputs.dot(self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs


iterations = 10000
learning_rate = 1.0
hidden_nodes = 6
output_nodes = 1
