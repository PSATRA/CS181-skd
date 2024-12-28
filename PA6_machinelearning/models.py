import nn
import numpy as np
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        node_value = nn.as_scalar(self.run(x))
        return 1 if node_value >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            converged = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    converged = False
                    self.w.update(x, nn.as_scalar(y))
            if converged:
                return

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 1e-2
        self.batch_size = 8

        self.input_size = 1
        self.hidden_1_size = 128
        self.hidden_2_size = 64
        self.output_size = 1

        self.w1 = nn.Parameter(self.input_size, self.hidden_1_size)
        self.w2 = nn.Parameter(self.hidden_1_size, self.hidden_2_size)
        self.w3 = nn.Parameter(self.hidden_2_size, self.output_size)

        self.b1 = nn.Parameter(1, self.hidden_1_size)
        self.b2 = nn.Parameter(1, self.hidden_2_size)
        self.b3 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.w1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w2)
        x = nn.AddBias(x, self.b2)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w3)
        x = nn.AddBias(x, self.b3)  # output layer
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SquareLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                self.w1.update(gradient[0], -self.lr)
                self.w2.update(gradient[1], -self.lr)
                self.w3.update(gradient[2], -self.lr)
                self.b1.update(gradient[3], -self.lr)
                self.b2.update(gradient[4], -self.lr)
                self.b3.update(gradient[5], -self.lr)
                if nn.as_scalar(loss) < 0.001:
                    return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 5e-2
        self.lr_decay = 0.6
        self.lr_minimum = 1e-3

        self.batch_size = 8

        self.input_size = 784
        self.hidden_1_size = 128
        self.hidden_2_size = 64
        self.output_size = 10

        self.w1 = nn.Parameter(self.input_size, self.hidden_1_size)
        self.w2 = nn.Parameter(self.hidden_1_size, self.hidden_2_size)
        self.w3 = nn.Parameter(self.hidden_2_size, self.output_size)

        self.b1 = nn.Parameter(1, self.hidden_1_size)
        self.b2 = nn.Parameter(1, self.hidden_2_size)
        self.b3 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.w1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w2)
        x = nn.AddBias(x, self.b2)
        x = nn.ReLU(x)

        x = nn.Linear(x, self.w3)
        x = nn.AddBias(x, self.b3)  # output layer
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                self.w1.update(gradient[0], -self.lr)
                self.w2.update(gradient[1], -self.lr)
                self.w3.update(gradient[2], -self.lr)
                self.b1.update(gradient[3], -self.lr)
                self.b2.update(gradient[4], -self.lr)
                self.b3.update(gradient[5], -self.lr)

            self.lr = max(self.lr_minimum, self.lr * self.lr_decay)
            if dataset.get_validation_accuracy() >= 0.975:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.lr = 1e-1
        self.lr_decay = 0.9
        self.lr_minimum = 1e-3

        self.batch_size = 8

        self.input_size = self.num_chars
        self.hidden_size = 512
        self.output_size = 5

        self.w1 = nn.Parameter(self.input_size, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.w3 = nn.Parameter(self.hidden_size, self.output_size)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(xs[0], self.w1)
        h = nn.ReLU(x)
        for x in xs[1:]:
            x = nn.Linear(x, self.w1)
            x = nn.ReLU(x)

            h = nn.Linear(h, self.w2)
            h = nn.ReLU(h)

            h = nn.Add(x, h)

        f = nn.Linear(h, self.w3)   # output layer
        return f

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(xs)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradient = nn.gradients(loss, [self.w1, self.w2, self.w3])
                self.w1.update(gradient[0], -self.lr)
                self.w2.update(gradient[1], -self.lr)
                self.w3.update(gradient[2], -self.lr)

            self.lr = max(self.lr_minimum, self.lr * self.lr_decay)
            if dataset.get_validation_accuracy() >= 0.85:
                return

class Attention(object):
    def __init__(self, layer_size, block_size):
        """
        Initializes the Attention layer.

        Arguments:
            layer_size: The dimensionality of the input and output vectors.
            block_size: The size of the block for the causal mask (used to apply causal attention).
        
        We initialize the weight matrices (K, Q, and V) using random normal distributions.
        The causal mask is a lower triangular matrix (a matrix of zeros above the diagonal, ones on and below the diagonal).
        """

        self.k_weight = np.random.randn(layer_size, layer_size)
        self.q_weight = np.random.randn(layer_size, layer_size)
        self.v_weight = np.random.randn(layer_size, layer_size)

        # Create the causal mask using numpy
        self.mask = np.tril(np.ones((block_size, block_size)))  
        
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to the input tensor. This includes computing the query, key, and value matrices,
        calculating the attention scores, applying the causal mask, and then generating the output.

        Arguments:
            input: The input tensor of shape (batch_size, block_size, layer_size).

        Returns:
            output: The output tensor after applying the attention mechanism to the input.
        
        Remark: remember to use the causal mask and nn.softmax (in nn.py) will be helpful.
        """
        
        B, T, C = input.shape
        
        """YOUR CODE HERE"""
        # Compute Query (Q), Key (K), and Value (V) matrices
        Q = np.matmul(input, self.q_weight)
        K = np.matmul(input, self.k_weight)
        V = np.matmul(input, self.v_weight)

        attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.layer_size)
        causal_mask = self.mask
        attention_scores = np.where(causal_mask, attention_scores, float('-inf'))
        attention_weights = nn.softmax(attention_scores, -1)
        output = np.matmul(attention_weights, V)
        return output