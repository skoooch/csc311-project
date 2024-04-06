
from sympy import true
from utils import *
import numpy as np
import torch

import matplotlib.pyplot as plt
from torch import sigmoid
from torch import relu
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
num_questions = 1774
num_subjects = 388
def load_data(base_path="../project/data", question = False):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    
    if question:
        train_matrix = train_matrix.transpose()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_input, g = 50, h = 50, question=False):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_input, g)
        self.h = nn.Linear(h, num_input)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encode = sigmoid(self.g(inputs))
        decode = sigmoid(self.h(encode))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decode

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, question=False):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    for epoch in range(0, num_epoch):
        train_loss = 0
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()
            optimizer.zero_grad()
            output = model(inputs)
            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            
            loss = torch.sum((output - target) ** 2) + model.get_weight_norm()*lamb*0.5
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        valid_acc = evaluate(model, zero_train_data, valid_data, question)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def evaluate(model, train_data, valid_data, question=False):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    if not question:
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(train_data[u]).unsqueeze(0)
            output = model(inputs)

            guess = output[0][valid_data["question_id"][i]].item() >= 0.5
            if guess == valid_data["is_correct"][i]:
                correct += 1
            total += 1
    else:
        for i, u in enumerate(valid_data["question_id"]):
            inputs = Variable(train_data[u]).unsqueeze(0)
            output = model(inputs)

            guess = output[0][valid_data["user_id"][i]].item() >= 0.5
            if guess == valid_data["is_correct"][i]:
                correct += 1
            total += 1
    return correct / float(total)


def main():
    question = False
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(question=question)
    model = AutoEncoder(train_matrix.shape[1], question=question)
    train(model, 0.1, 0.001,  train_matrix,  zero_train_matrix, valid_data, 100, question=question)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
