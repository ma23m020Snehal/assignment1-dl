#!/usr/bin/env python
import argparse
import wandb
import numpy as np
import math
from keras.datasets import fashion_mnist, mnist

class Optimizer:
    # Base optimizer class to handle initialization of parameters.
    def __init__(self, lr):  # where lr:learning rate
        self.lr = lr

    def update(sts, biases, dW, db, t):
        raise NotImplementedError("Subclasses should implement this method.")


class SGD(Optimizer):
    # Stochastic Gradient Descent Optimizer.
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, wts,bias, dW, db, t):
        for i in range(len(wts)):
            wts[i] -= self.lr * dW[i]
            bias[i]  -= self.lr * db[i]


class Momentum(Optimizer):
    # Momentum-based Gradient Descent.
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.state = {}

    def update(self, wts, bias, dW, db, t):
        if not self.state:
            self.state['vw'] = [np.zeros_like(w) for w in wts]
            self.state['vb'] = [np.zeros_like(b) for b in bias]

        for i in range(len(wts)):
            self.state['vw'][i] = self.momentum * self.state['vw'][i] + self.lr * dW[i]
            self.state['vb'][i] = self.momentum * self.state['vb'][i] + self.lr * db[i]
            wts[i] -= self.state['vw'][i]
            bias[i]  -= self.state['vb'][i]


class Nesterov(Momentum):
    # Nesterov Accelerated Gradient Descent.
    def update(self, wts, bias, dW, db, t):
        if not self.state:
            self.state['vw'] = [np.zeros_like(w) for w in wts]
            self.state['vb'] = [np.zeros_like(b) for b in bias]

        for i in range(len(wts)):
            prev_vw = self.state['vw'][i].copy()
            prev_vb = self.state['vb'][i].copy()

            self.state['vw'][i] = self.momentum * self.state['vw'][i] + self.lr * dW[i]
            self.state['vb'][i] = self.momentum * self.state['vb'][i] + self.lr * db[i]

            wts[i] -= -self.momentum * prev_vw + (1 + self.momentum) * self.state['vw'][i]
            bias[i]  -= -self.momentum * prev_vb + (1 + self.momentum) * self.state['vb'][i]


class RMSProp(Optimizer):
    # RMSProp Optimizer
    def __init__(self, lr=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.state = {}

    def update(self, wts, bias, dW, db, t):
        if not self.state:
            self.state['Sw'] = [np.zeros_like(w) for w in wts]
            self.state['Sb'] = [np.zeros_like(b) for b in bias]

        for i in range(len(wts)):
            self.state['Sw'][i] = self.decay_rate * self.state['Sw'][i] + (1 - self.decay_rate) * (dW[i] ** 2)
            self.state['Sb'][i] = self.decay_rate * self.state['Sb'][i] + (1 - self.decay_rate) * (db[i] ** 2)

            wts[i] -= self.lr * dW[i] / (np.sqrt(self.state['Sw'][i]) + self.epsilon)
            bias[i]  -= self.lr * db[i] / (np.sqrt(self.state['Sb'][i]) + self.epsilon)


class Adam(Optimizer):
    # Adam Optimizer
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {}

    def update(self, wts, bias, dW, db, t):
        if not self.state:
            self.state['mw'] = [np.zeros_like(w) for w in wts]
            self.state['vw'] = [np.zeros_like(w) for w in wts]
            self.state['mb'] = [np.zeros_like(b) for b in bias]
            self.state['vb'] = [np.zeros_like(b) for b in bias]

        for i in range(len(wts)):
            self.state['mw'][i] = self.beta1 * self.state['mw'][i] + (1 - self.beta1) * dW[i]
            self.state['vw'][i] = self.beta2 * self.state['vw'][i] + (1 - self.beta2) * (dW[i] ** 2)
            self.state['mb'][i] = self.beta1 * self.state['mb'][i] + (1 - self.beta1) * db[i]
            self.state['vb'][i] = self.beta2 * self.state['vb'][i] + (1 - self.beta2) * (db[i] ** 2)

            m_hat_w = self.state['mw'][i] / (1 - self.beta1 ** t)
            v_hat_w = self.state['vw'][i] / (1 - self.beta2 ** t)
            m_hat_b = self.state['mb'][i] / (1 - self.beta1 ** t)
            v_hat_b = self.state['vb'][i] / (1 - self.beta2 ** t)

            wts[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            bias[i]  -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Nadam(Adam):
    # Nadam Optimizer (Adam with Nesterov momentum).
    def update(self, wts, bias, dW, db, t):
        if not self.state:
            self.state['mw'] = [np.zeros_like(w) for w in wts]
            self.state['vw'] = [np.zeros_like(w) for w in wts]
            self.state['mb'] = [np.zeros_like(b) for b in bias]
            self.state['vb'] = [np.zeros_like(b) for b in bias]

        for i in range(len(wts)):
            self.state['mw'][i] = self.beta1 * self.state['mw'][i] + (1 - self.beta1) * dW[i]
            self.state['vw'][i] = self.beta2 * self.state['vw'][i] + (1 - self.beta2) * (dW[i] ** 2)
            self.state['mb'][i] = self.beta1 * self.state['mb'][i] + (1 - self.beta1) * db[i]
            self.state['vb'][i] = self.beta2 * self.state['vb'][i] + (1 - self.beta2) * (db[i] ** 2)

            m_hat = self.state['mw'][i] / (1 - self.beta1 ** t)
            v_hat = self.state['vw'][i] / (1 - self.beta2 ** t)
            m_hat_b = self.state['mb'][i] / (1 - self.beta1 ** t)
            v_hat_b = self.state['vb'][i] / (1 - self.beta2 ** t)

            wts[i] -= self.lr * (self.beta1 * m_hat + (1 - self.beta1) * dW[i] / (1 - self.beta1 ** t)) / (np.sqrt(v_hat) + self.epsilon)
            bias[i]  -= self.lr * (self.beta1 * m_hat_b + (1 - self.beta1) * db[i] / (1 - self.beta1 ** t)) / (np.sqrt(v_hat_b) + self.epsilon)

optimizers_dicto = {
    "sgd": SGD,
    "momentum":  Momentum,
    "nag":  Nesterov,
    "rmsprop":  RMSProp,
    "adam":  Adam,
    "nadam":  Nadam
}
def xavier_init(fan_in, fan_out):
    """Xavier/Glorot initialization."""
    bound = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, (fan_in, fan_out))


class MLPmodel:
    def __init__(self, input_dim, num_hidden_layers, hidden_size, output_dim, weight_init='random', activation='relu', weight_decay=0.0, loss_type='cross_entropy'):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        
        # Set activation functions
        if activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1.0 / (1.0 + np.exp(-x))
            self.activation_deriv = lambda x: self.activation(x) * (1 - self.activation(x))
        elif activation == 'tanh':
            self.activation = lambda x: np.tanh(x)
            self.activation_deriv = lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError("Unsupported activation")
        
        # Build network architecture
        layer_dims = [input_dim] + [hidden_size]*num_hidden_layers + [output_dim]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_dims)-1):
            fan_in  = layer_dims[i]
            fan_out = layer_dims[i+1]
            if weight_init == 'xavier':
                W = xavier_init(fan_in, fan_out)
            else:
                W = np.random.randn(fan_in, fan_out) * 0.01
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        self.zs = []           # store linear combinations
        self.activations = [X]  # input activation
        A = X
        for i in range(len(self.weights)-1):
            z = A.dot(self.weights[i]) + self.biases[i]
            self.zs.append(z)
            A = self.activation(z)
            self.activations.append(A)
        # Output layer
        z = A.dot(self.weights[-1]) + self.biases[-1]
        self.zs.append(z)
        out = self.softmax(z)
        self.activations.append(out)
        return out
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_type == 'cross_entropy':
            loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        elif self.loss_type == 'squared_error':
            loss = 0.5 * np.sum((y_pred - y_true)**2) / m
        if self.weight_decay > 0:
            for W in self.weights:
                loss += (self.weight_decay / (2*m)) * np.sum(W**2)
        return loss
    
    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        # Compute delta for output layer
        if self.loss_type == 'cross_entropy':
            delta = (y_pred - y_true) / m
        elif self.loss_type == 'squared_error':
            delta = (y_pred - y_true) / m  # approximate gradient for squared error
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        
        # Output layer gradients
        dW[-1] = self.activations[-2].T.dot(delta)
        db[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta = delta.dot(self.weights[i+1].T) * self.activation_deriv(self.zs[i])
            dW[i] = self.activations[i].T.dot(delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
        
        if self.weight_decay > 0:
            for i in range(len(self.weights)):
                dW[i] += (self.weight_decay * self.weights[i]) / m
        
        return dW, db


def load_dataset(dataset_name):
    if dataset_name.lower() == "mnist":
        return mnist.load_data()
    else:
        return fashion_mnist.load_data()

def preprocess_data(train_images, train_labels_mnist, test_images, test_labels):
    val_split = int(0.1 * train_images.shape[0])
    val_images = train_images[:val_split]
    val_labels_mnist = train_labels_mnist[:val_split]
    train_images = train_images[val_split:]
    train_labels_mnist = train_labels_mnist[val_split:]

    train_images = train_images.reshape(-1, 28*28) / 255.0
    val_images = val_images.reshape(-1, 28*28) / 255.0
    test_images = test_images.reshape(-1, 28*28) / 255.0
    
    train_labels = np.eye(10)[train_labels_mnist]
    val_labels = np.eye(10)[val_labels_mnist]
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels, val_labels_mnist

def initialize_model(args):
    return MLPmodel(
        input_dim=784,
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_dim=10,
        weight_init=args.weight_init,
        activation=args.activation,
        weight_decay=args.weight_decay,
        loss_type=args.loss
    )

def initialize_optimizer(args):
    opt_class = optimizers_dicto.get(args.optimizer.lower())
    if opt_class is None:
        raise ValueError(f"Unsupported optimizer : {args.optimizer}")
    if args.optimizer.lower() == "sgd":
        return opt_class(lr=args.learning_rate)
    elif args.optimizer.lower() in ["momentum", "nag"]:
        return opt_class(lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer.lower() == "rmsprop":
        return opt_class(lr=args.learning_rate, beta=args.beta, epsilon=args.epsilon)
    elif args.optimizer.lower() in ["adam", "nadam"]:
        return opt_class(lr=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    else:
        return opt_class(lr=args.learning_rate)

def train_model(model, optimizer, train_images, train_labels, val_images, val_labels, val_labels_mnist, args):
    num_samples = train_images.shape[0]
    num_batches = num_samples // args.batch_size
    global_step = 1
    
    for epoch in range(args.epochs):
        indices = np.random.permutation(num_samples)
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]
        epoch_loss = 0.0
        
        for i in range(num_batches):
            start = i * args.batch_size
            end = start + args.batch_size
            X_batch = train_images_shuffled[start:end]
            y_batch = train_labels_shuffled[start:end]
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            dW, db = model.backward(y_pred, y_batch)
            optimizer.update(model.weights, model.biases, dW, db, global_step)
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        val_acc, val_loss = evaluate_model(model, val_images, val_labels, val_labels_mnist)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "val_loss": val_loss, "val_accuracy": val_acc})

def evaluate_model(model, images, labels, true_labels):
    predictions = model.forward(images)
    loss = model.compute_loss(predictions, labels)
    accuracy = np.mean(np.argmax(predictions, axis=1) == true_labels)
    return accuracy, loss

def train_image_classifier(args):
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))
    
    (train_images, train_labels_mnist), (test_images, test_labels) = load_dataset(args.dataset)
    train_images, train_labels, val_images, val_labels, test_images, test_labels, val_labels_mnist = preprocess_data(train_images, train_labels_mnist, test_images, test_labels)
    
    model = initialize_model(args)
    optimizer = initialize_optimizer(args)
    
    train_model(model, optimizer, train_images, train_labels, val_images, val_labels, val_labels_mnist, args)
    
    test_acc, _ = evaluate_model(model, test_images, np.eye(10)[test_labels], test_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()

# Main: Argument Parsing and Script Execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function.")
    parser.add_argument("-o", "--optimizer", default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", default="Xavier", choices=["random", "Xavier"], help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", default="relu", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function.")
    
    args = parser.parse_args()
    train_image_classifier(args)