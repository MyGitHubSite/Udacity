import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        #   Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        #   Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))  
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        'TODO: Set self.activation_function to your implemented sigmoid function'

        # Activation Functions
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))                               # sigmoid

    def train(self, features, targets):                 #   Train the network on batch of features and targets. 

        #   features: 2D array, each row is one data record, each column is a feature
        #   targets:  1D array of target values

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        #  print ("# delta_weights_i_h: ", delta_weights_i_h.shape)                     # delta_weights_i_h:  (3, 2)
        #  print ("# delta_weights_h_o: ", delta_weights_h_o.shape)                     # delta_weights_h_o:  (2, 1)
        
        for X, y in zip(features, targets):
            
        #   Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  

        #   Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):                    #   Implement the forward pass here

        # X: features batch
        #  print (X)
        #  print (self.weights_input_to_hidden)    
        #  print ("# X: ", X.shape)                                                               # X:  (3,)
        #  print ("# X[]: ", X[:,None].shape)                                                     # X[]:  (3, 1)
        #  print ("# weights_input_to_hidden: ", self.weights_input_to_hidden.shape)              # weights_input_to_hidden:  (3, 2)
        
        '''TODO'''
        hidden_inputs = np.dot(self.weights_input_to_hidden.T, X[:,None])  

        #  print ("# hidden_inputs: ", hidden_inputs.shape)                                       # hidden_inputs:  (2, 1)       
        hidden_outputs = self.activation_function(hidden_inputs)

        #  print ("# hidden_outputs: ", hidden_outputs.shape)                                     # hidden_outputs:  (2, 1)               
        #  print ("# weights_hidden_to_output: ", self.weights_hidden_to_output.shape)            # weights_hidden_to_output:  (2, 1)
        final_inputs = np.dot(self.weights_hidden_to_output.T, hidden_outputs)

        #  print ("# final_inputs: ", final_inputs.shape)                                         # final_inputs:  (1, 1)
        final_outputs = final_inputs                                      

        #  print ("# final_outputs: ", final_outputs.shape)                                       # final_outputs:  (1, 1)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        #   final_outputs: output from forward pass
        #   y: target (i.e. label) batch
        #   delta_weights_i_h: change in weights from input to hidden layers
        #   delta_weights_h_o: change in weights from hidden to output layers

        #  print ("# y: ", y.shape)                                                     # y:  (1,)
        #  print ("# y: ", y[:,None].shape)                                             # y:  (1, 1)     
        #  print ("# delta_weights_i_h: ", delta_weights_i_h.shape)                     # delta_weights_i_h:  (3, 2)
        #  print ("# delta_weights_h_o: ", delta_weights_h_o.shape)                     # delta_weights_h_o:  (2, 1)
               
        ''' TODO '''
        error = y - final_outputs                                                       # output layer error is target less actual'

        #  print ("# error: ", error.shape)                                             # error:  (1, 1)
        output_error_term = error                                                       # backprograted error term'

        #  print ("# output_error_term: ", output_error_term.shape)                     # output_error_term:  (1, 1)
        #  print ("# weights_hidden_to_output: ", self.weights_hidden_to_output.shape)  # weights_hidden_to_output:  (2, 1)       
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)       

        #  print ("# hidden_error: ", hidden_error.shape)                               # hidden_error:  (1, 2)     
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)        # backprograted error term

        #  print ("# hidden_error_term: ", hidden_error_term.shape)                     # hidden_error_term:  (1, 2)
        
        delta_weights_i_h += np.dot(X[:,None], hidden_error_term.T)                     # i_h weight step
        delta_weights_h_o += np.dot(hidden_outputs, error)                              # h_o weight step
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):          # Update weights on gradient descent step
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records 
        # update input-to-hidden weights with gradient descent step '''
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records            

    def run(self, features):                            # Run a forward pass through the network with input features

        #   features: 1D array of feature values
        ''' TODO '''
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)          # X: 1x3 W1: 3x2 -> 1x2 signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)                # 1x2 signals from hidden layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)    # HO: 1x2  W2: 2x1 -> 1x1 signals into final output layer
        final_outputs = final_inputs                                            # xx1 signals from final output layer
        
        return final_outputs

# Set your hyperparameters here

iterations = 500000
learning_rate = 0.5
hidden_nodes = 20
output_nodes = 1
