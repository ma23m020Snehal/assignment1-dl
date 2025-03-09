#!/usr/bin/env python
import argparse
import wandb
import numpy as np
import math
from keras.datasets import fashion_mnist, mnist


# Optimizers


class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    def update(self, weights, biases, dW, db, t):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * dW[i]
            biases[i]  -= self.learning_rate * db[i]


class Momentum:
    def __init__(self, learning_rate=0.1, momentum=0.5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vw = None
        self.vb = None
    def update(self, weights, biases, dW, db, t):
        if self.vw is None:
            self.vw = [np.zeros_like(w) for w in weights]
            self.vb = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.vw[i] = self.momentum * self.vw[i] + self.learning_rate * dW[i]
            self.vb[i] = self.momentum * self.vb[i] + self.learning_rate * db[i]
            weights[i] -= self.vw[i]
            biases[i]  -= self.vb[i]


class Nesterov:           # "nag"
    def __init__(self, learning_rate=0.1, momentum=0.5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vw = None
        self.vb = None
    def update(self, weights, biases, dW, db, t):
        if self.vw is None:
            self.vw = [np.zeros_like(w) for w in weights]
            self.vb = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            v_prev_w = self.vw[i].copy()
            v_prev_b = self.vb[i].copy()
            self.vw[i] = self.momentum * self.vw[i] + self.learning_rate * dW[i]
            self.vb[i] = self.momentum * self.vb[i] + self.learning_rate * db[i]
            weights[i] -= -self.momentum * v_prev_w + (1 + self.momentum) * self.vw[i]
            biases[i]  -= -self.momentum * v_prev_b + (1 + self.momentum) * self.vb[i]



class RMSProp:
    def __init__(self, learning_rate=0.1, beta=0.5, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.Sw = None
        self.Sb = None
    def update(self, weights, biases, dW, db, t):
        if self.Sw is None:
            self.Sw = [np.zeros_like(w) for w in weights]
            self.Sb = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.Sw[i] = self.beta * self.Sw[i] + (1 - self.beta) * (dW[i] ** 2)
            self.Sb[i] = self.beta * self.Sb[i] + (1 - self.beta) * (db[i] ** 2)
            weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.Sw[i]) + self.epsilon)
            biases[i]  -= self.learning_rate * db[i] / (np.sqrt(self.Sb[i]) + self.epsilon)


class Adam:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mw = None
        self.vw = None
        self.mb = None
        self.vb = None
    def update(self, weights, biases, dW, db, t):
        if self.mw is None:
            self.mw = [np.zeros_like(w) for w in weights]
            self.vw = [np.zeros_like(w) for w in weights]
            self.mb = [np.zeros_like(b) for b in biases]
            self.vb = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * dW[i]
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db[i]
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (db[i] ** 2)
            m_hat_w = self.mw[i] / (1 - self.beta1 ** t)
            v_hat_w = self.vw[i] / (1 - self.beta2 ** t)
            m_hat_b = self.mb[i] / (1 - self.beta1 ** t)
            v_hat_b = self.vb[i] / (1 - self.beta2 ** t)
            weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i]  -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Nadam:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mw = None
        self.vw = None
        self.mb = None
        self.vb = None
    def update(self, weights, biases, dW, db, t):
        if self.mw is None:
            self.mw = [np.zeros_like(w) for w in weights]
            self.vw = [np.zeros_like(w) for w in weights]
            self.mb = [np.zeros_like(b) for b in biases]
            self.vb = [np.zeros_like(b) for b in biases]
        for i in range(len(weights)):
            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * dW[i]
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db[i]
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (db[i] ** 2)
            m_hat = self.mw[i] / (1 - self.beta1 ** t)
            v_hat = self.vw[i] / (1 - self.beta2 ** t)
            m_hat_b = self.mb[i] / (1 - self.beta1 ** t)
            v_hat_b = self.vb[i] / (1 - self.beta2 ** t)
            weights[i] -= self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * dW[i] / (1 - self.beta1 ** t)) / (np.sqrt(v_hat) + self.epsilon)
            biases[i]  -= self.learning_rate * (self.beta1 * m_hat_b + (1 - self.beta1) * db[i] / (1 - self.beta1 ** t)) / (np.sqrt(v_hat_b) + self.epsilon)


# Mapping from optimizer string to class
optimizer_dict = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": Nesterov,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": Nadam
}


# Model Definition: MLP with Loss Function

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

class SimpleMLPWithLoss:
    def __init__(self, input_dim, num_hidden_layers, hidden_size, output_dim,
                 weight_init='random', activation='sigmoid', weight_decay=0.0, loss_type='cross_entropy'):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        
        # Set activation function and its derivative
        act = activation.lower()
        if act == "relu":
            self.activation = lambda x: np.maximum(0, x)
            self.activation_deriv = lambda x: (x > 0).astype(float)
        elif act == "sigmoid":
            self.activation = lambda x: 1.0/(1.0+np.exp(-x))
            self.activation_deriv = lambda x: self.activation(x) * (1 - self.activation(x))
        elif act == "tanh":
            self.activation = lambda x: np.tanh(x)
            self.activation_deriv = lambda x: 1 - np.tanh(x)**2
        elif act == "identity":
            self.activation = lambda x: x
            self.activation_deriv = lambda x: np.ones_like(x)
        else:
            raise ValueError("Unsupported activation function")
        
        # Build network architecture
        layer_dims = [input_dim] + [hidden_size]*num_hidden_layers + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_dims)-1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i+1]
            if weight_init.lower() == "xavier":
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
        self.zs = []
        self.activations = [X]
        A = X
        for i in range(len(self.weights)-1):
            z = A.dot(self.weights[i]) + self.biases[i]
            self.zs.append(z)
            A = self.activation(z)
            self.activations.append(A)
        z = A.dot(self.weights[-1]) + self.biases[-1]
        self.zs.append(z)
        out = self.softmax(z)
        self.activations.append(out)
        return out
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_type == "cross_entropy":
            loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        elif self.loss_type == "mean_squared_error":
            loss = 0.5 * np.sum((y_pred - y_true)**2) / m
        else:
            raise ValueError("Unsupported loss type")
        if self.weight_decay > 0:
            for W in self.weights:
                loss += (self.weight_decay / (2*m)) * np.sum(W**2)
        return loss
    
    def backward(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_type == "cross_entropy":
            delta = (y_pred - y_true) / m
        elif self.loss_type == "mean_squared_error":
            delta = (y_pred - y_true) / m  # approximate gradient
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        dW[-1] = self.activations[-2].T.dot(delta)
        db[-1] = np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.weights)-2, -1, -1):
            delta = delta.dot(self.weights[i+1].T) * self.activation_deriv(self.zs[i])
            dW[i] = self.activations[i].T.dot(delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
        if self.weight_decay > 0:
            for i in range(len(self.weights)):
                dW[i] += (self.weight_decay * self.weights[i]) / m
        return dW, db


# Training Function

def train_model(args):
    # Initialize wandb run
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))
    
    # Load dataset
    if args.dataset.lower() == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    else:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    
    # Split off 10% of training data for validation
    val_split = int(0.1 * X_train_full.shape[0])
    X_val = X_train_full[:val_split]
    y_val = y_train_full[:val_split]
    X_train = X_train_full[val_split:]
    y_train = y_train_full[val_split:]
    
    # Preprocess: flatten and normalize
    X_train = X_train.reshape(-1, 28*28) / 255.0
    X_val   = X_val.reshape(-1, 28*28) / 255.0
    X_test  = X_test.reshape(-1, 28*28) / 255.0
    
    # One-hot encode for training and validation
    def one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]
    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)
    
    # Build the model
    model = SimpleMLPWithLoss(
        input_dim=784,
        num_hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_dim=10,
        weight_init=args.weight_init,
        activation=args.activation,
        weight_decay=args.weight_decay,
        loss_type=args.loss
    )
    
    # Choose optimizer
    opt_class = optimizer_dict.get(args.optimizer.lower())
    if opt_class is None:
        raise ValueError("Unsupported optimizer")
    if args.optimizer.lower() in ["sgd"]:
        optimizer = opt_class(learning_rate=args.learning_rate)
    elif args.optimizer.lower() in ["momentum", "nag"]:
        optimizer = opt_class(learning_rate=args.learning_rate, momentum=args.momentum)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = opt_class(learning_rate=args.learning_rate, beta=args.beta, epsilon=args.epsilon)
    elif args.optimizer.lower() in ["adam", "nadam"]:
        optimizer = opt_class(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    else:
        optimizer = opt_class(learning_rate=args.learning_rate)
    
    # Training loop
    num_samples = X_train.shape[0]
    num_batches = num_samples // args.batch_size
    global_step = 1
    for epoch in range(args.epochs):
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_oh[indices]
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * args.batch_size
            end = start + args.batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            dW, db = model.backward(y_pred, y_batch)
            optimizer.update(model.weights, model.biases, dW, db, global_step)
            global_step += 1
        avg_loss = epoch_loss / num_batches
        
        # Evaluate on validation set
        val_pred = model.forward(X_val)
        val_loss = model.compute_loss(val_pred, y_val_oh)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    
    # Evaluate on test set
    test_pred = model.forward(X_test)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    wandb.finish()


# Main: Argument Parsing and Script Execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feedforward neural network with wandb logging.")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function.")
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", default="Xavier", choices=["random", "Xavier"], help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", default="ReLU", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function.")
    
    args = parser.parse_args()
    train_model(args)
