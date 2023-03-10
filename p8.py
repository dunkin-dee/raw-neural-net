import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class Layer_Dense():
  def __init__(self, n_inputs, n_neurons,
               weight_regularizer_l1=0, weight_regularizer_l2=0,
               bias_regularizer_l1=0, bias_regularizer_l2=0) -> None:
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
    #set regularization strength
    self.weight_regularizer_l1 = weight_regularizer_l1
    self.weight_regularizer_l2 = weight_regularizer_l2
    self.bias_regularizer_l1 = bias_regularizer_l1
    self.bias_regularizer_l2 = bias_regularizer_l2

  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    self.inputs = inputs
  def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU():
  def __init__(self) -> None:
      pass
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)
    self.inputs = inputs
  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0



class Activation_Softmax:
  def __init__(self) -> None:
    pass
  def forward(self, inputs):
    self.inputs = inputs
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values/ np.sum(exp_values, axis=1, keepdims=True)

  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1, 1)
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)



class Loss:
  # Regularization loss calculation
  def regularization_loss(self, layer):
  # 0 by default
    regularization_loss = 0
    # L1 regularization - weights
    # calculate only when factor greater than 0
    if layer.weight_regularizer_l1 > 0:
      regularization_loss += layer.weight_regularizer_l1 * \
                          np.sum(np.abs(layer.weights))
    # L2 regularization - weights
    if layer.weight_regularizer_l2 > 0:
      regularization_loss += layer.weight_regularizer_l2 * \
                          np.sum(layer.weights * \
                          layer.weights)

    # L1 regularization - biases
    # calculate only when factor greater than 0
    if layer.bias_regularizer_l1 > 0:
      regularization_loss += layer.bias_regularizer_l1 * \
                          np.sum(np.abs(layer.biases))
    # L2 regularization - biases
    if layer.bias_regularizer_l2 > 0:
      regularization_loss += layer.bias_regularizer_l2 * \
                            np.sum(layer.biases * \
                            layer.biases)
    return regularization_loss
  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):
    # Calculate sample losses
    sample_losses = self.forward(output, y)
    # Calculate mean loss
    data_loss = np.mean(sample_losses)
    # Return loss
    return data_loss


class Loss_CategoricalCrossEntropy(Loss):
  def forward(self, y_preds, y_true):
    samples = len(y_preds)
    y_preds_clipped = np.clip(y_preds, 1e-7, 1-1e-7)

    if len(y_true.shape) == 1:
      correct_confidences = y_preds_clipped[range(samples), y_true]
    elif len(y_true.shape) == 2:
      correct_confidences = y_preds_clipped[range(samples), np.argmax(y_true, axis=1)]
  
    neg_log_likelihoods = -np.log(correct_confidences)
    return(neg_log_likelihoods)

  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    labels = len(dvalues[0])
    if len(y_true.shape) == 1:
      y_true = np.eye(labels)[y_true]
    self.dinputs = -y_true/dvalues
    self.dinputs = self.dinputs/samples






class Activation_Softmax_Loss_CategoricalCrossentropy():
  def __init__(self) -> None:
    self.activation = Activation_Softmax()
    self.loss = Loss_CategoricalCrossEntropy()

  def forward(self, inputs, y_true):
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)

  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)

    self.dinputs = dvalues.copy()
    self.dinputs[range(samples), y_true] -= 1
    self.dinputs = self.dinputs/samples


# SGD optimizer
class Optimizer_SGD:
  # Initialize optimizer - set settings,
  # learning rate of 1. is default for this optimizer
  def __init__(self, learning_rate=1., decay=0., momentum=0.):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum
  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
        (1. / (1. + self.decay * self.iterations))
  
  # Update parameters
  def update_params(self, layer):
    # If we use momentum
    if self.momentum:
      # If layer does not contain momentum arrays, create them
      # filled with zeros
      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)
        # If there is no momentum array for weights
        # The array doesn't exist for biases yet either.
        layer.bias_momentums = np.zeros_like(layer.biases)
      # Build weight updates with momentum - take previous
      # updates multiplied by retain factor and update with
      # current gradients
      weight_updates = \
        self.momentum * layer.weight_momentums - \
        self.current_learning_rate * layer.dweights
      layer.weight_momentums = weight_updates
      # Build bias updates
      bias_updates = \
        self.momentum * layer.bias_momentums - \
        self.current_learning_rate * layer.dbiases
      layer.bias_momentums = bias_updates
      # Vanilla SGD updates (as before momentum update)
    else:
      weight_updates = -self.current_learning_rate * \
        layer.dweights
      bias_updates = -self.current_learning_rate * \
        layer.dbiases
    # Update weights and biases using either
    # vanilla or momentum updates
    layer.weights += weight_updates
    layer.biases += bias_updates
  
  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1



# Adagrad optimizer
class Optimizer_Adagrad:
  # Initialize optimizer - set settings
  def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
      (1. / (1. + self.decay * self.iterations))
      # Update parameters
  def update_params(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
    # Update cache with squared current gradients
    layer.weight_cache += layer.dweights**2
    layer.bias_cache += layer.dbiases**2
    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weights += -self.current_learning_rate * \
                      layer.dweights / \
                      (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += -self.current_learning_rate * \
                      layer.dbiases / \
                      (np.sqrt(layer.bias_cache) + self.epsilon)
  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1


# RMSprop optimizer
class Optimizer_RMSprop:
  # Initialize optimizer - set settings
  def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
  rho=0.9):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.rho = rho
  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
      (1. / (1. + self.decay * self.iterations))
  # Update parameters
  def update_params(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
    # Update cache with squared current gradients
    layer.weight_cache = self.rho * layer.weight_cache + \
      (1 - self.rho) * layer.dweights**2
    layer.bias_cache = self.rho * layer.bias_cache + \
      (1 - self.rho) * layer.dbiases**2
    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weights += -self.current_learning_rate * \
                      layer.dweights / \
                      (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += -self.current_learning_rate * \
                      layer.dbiases / \
                      (np.sqrt(layer.bias_cache) + self.epsilon)
  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1


class Optimizer_Adam:
  # Initialize optimizer - set settings
  def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
               beta_1=0.9, beta_2=0.999):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.epsilon = epsilon
    self.beta_1 = beta_1
    self.beta_2 = beta_2
  # Call once before any parameter updates
  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
        (1. / (1. + self.decay * self.iterations))
  # Update parameters
  def update_params(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_momentums = np.zeros_like(layer.weights)
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_momentums = np.zeros_like(layer.biases)
      layer.bias_cache = np.zeros_like(layer.biases)
    # Update momentum with current gradients
    layer.weight_momentums = self.beta_1 * \
                              layer.weight_momentums + \
                              (1 - self.beta_1) * layer.dweights
    layer.bias_momentums = self.beta_1 * \
                            layer.bias_momentums + \
                            (1 - self.beta_1) * layer.dbiases
    # Get corrected momentum
    # self.iteration is 0 at first pass
    # and we need to start with 1 here
    weight_momentums_corrected = layer.weight_momentums / \
                                (1 - self.beta_1 ** (self.iterations + 1))
    bias_momentums_corrected = layer.bias_momentums / \
                                (1 - self.beta_1 ** (self.iterations + 1))
    # Update cache with squared current gradients
    layer.weight_cache = self.beta_2 * layer.weight_cache + \
      (1 - self.beta_2) * layer.dweights**2
    layer.bias_cache = self.beta_2 * layer.bias_cache + \
      (1 - self.beta_2) * layer.dbiases**2

    # Get corrected cache
    weight_cache_corrected = layer.weight_cache / \
                            (1 - self.beta_2 ** (self.iterations + 1))
    bias_cache_corrected = layer.bias_cache / \
                            (1 - self.beta_2 ** (self.iterations + 1))
    # Vanilla SGD parameter update + normalization
    # with square rooted cache
    layer.weights += -self.current_learning_rate * \
                      weight_momentums_corrected / \
                      (np.sqrt(weight_cache_corrected) +
                      self.epsilon)
    layer.biases += -self.current_learning_rate * \
                    bias_momentums_corrected / \
                    (np.sqrt(bias_cache_corrected) +
                    self.epsilon)
  # Call once after any parameter updates
  def post_update_params(self):
    self.iterations += 1

# Create dataset
X, y = spiral_data(samples=1000, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                             bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(512, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)


# Train in loop
for epoch in range(10001):
  # Perform a forward pass of our training data through this layer
  dense1.forward(X)
  # Perform a forward pass through activation function
  # takes the output of first dense layer here
  activation1.forward(dense1.output)
  # Perform a forward pass through second Dense layer
  # takes outputs of activation function of first layer as inputs
  dense2.forward(activation1.output)
  # Perform a forward pass through the activation/loss function
  # takes the output of second dense layer here and returns loss
  data_loss = loss_activation.forward(dense2.output, y)

  # Calculate regularization penalty
  regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
                        loss_activation.loss.regularization_loss(dense2)
  # Calculate overall loss
  loss = data_loss + regularization_loss

  # Calculate accuracy from output of activation2 and targets
  # calculate values along first axis
  predictions = np.argmax(loss_activation.output, axis=1)
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  accuracy = np.mean(predictions==y)
  
  if not epoch % 100:
    print(f'epoch: {epoch}, ' +
          f'acc: {accuracy:.3f}, ' +
          f'loss: {loss:.3f}, ' +
          f'data_loss: {data_loss:.3f}, ' +
          f'reg_loss: {regularization_loss:.3f}, ' +
          f'lr: {optimizer.current_learning_rate}')
  # Backward pass
  loss_activation.backward(loss_activation.output, y)
  dense2.backward(loss_activation.dinputs)
  activation1.backward(dense2.dinputs)
  dense1.backward(activation1.dinputs)
  # Update weights and biases
  optimizer.pre_update_params()
  optimizer.update_params(dense1)
  optimizer.update_params(dense2)
  optimizer.post_update_params()

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
  y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')