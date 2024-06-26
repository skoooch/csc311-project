"""
This file has a modified version of the AutoEncoder where all modifications are used (question as entity, deeper network, dense refeeding, dropout, meta data injection).
"""
from sympy import true
from utils import *
import numpy as np
import torch
from IRT_injection import get_thetas
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
num_students = 542
def load_data(base_path="../data", question = False):
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
    if question:
        meta = torch.FloatTensor(load_question_meta(1774, 388, root_dir=base_path))
    else:
        meta_matrix = load_student_meta(num_students, root_dir=base_path)
        zero_meta_matrix = meta_matrix.copy()
        zero_meta_matrix[np.isnan(meta_matrix)] = 0
        meta = torch.FloatTensor(zero_meta_matrix)
    return zero_train_matrix, train_matrix, valid_data, test_data, meta


class AutoEncoder(nn.Module):
    def __init__(self, num_input, g = 50, h = 50, bottleneck = 25, meta_bottleneck = 5, question=False):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()
        self.meta = nn.Linear(num_subjects, meta_bottleneck)
        # Define linear functions.
        self.dp = nn.Dropout(0.8)
        self.g = nn.Linear(num_input, g)
        self.g_two = nn.Linear(g, bottleneck)
        self.h_two = nn.Linear(meta_bottleneck + bottleneck + 1, h)
        self.h = nn.Linear(h, num_input)
        

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        g_2_w_norm = torch.norm(self.g_two.weight, 2) ** 2
        h_2_w_norm = torch.norm(self.h_two.weight, 2) ** 2
        meta_norm = torch.norm(self.meta.weight, 2) ** 2
        return g_w_norm + h_w_norm + meta_norm + g_2_w_norm + h_2_w_norm

    def forward(self, inputs, meta, beta):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encode_students = self.dp(sigmoid(self.g_two(self.dp(sigmoid(self.g(inputs))))))
        encode_meta = (sigmoid(self.meta(meta)))
        betas = torch.FloatTensor([beta]).unsqueeze(0)
        encode = torch.cat((encode_students, encode_meta, betas), dim=1)
        decode = sigmoid(self.h(self.dp(sigmoid(self.h_two(self.dp(encode))))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decode

def train(model, lr, lamb, beta, question_meta, train_data, zero_train_data, valid_data, num_epoch, question=False):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param question_meta: 2D FloatTensor
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    
    # Tell PyTorch you are training the model.
    model.train()
    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_question = train_data.shape[0]
    valid_array = []
    train_array = []
    epoch_array = []
    for epoch in range(0, num_epoch):
        train_loss = 0
        for question_id in range(num_question):
            inputs = Variable(zero_train_data[question_id]).unsqueeze(0)
            meta = question_meta[question_id].unsqueeze(0)
            target = inputs.clone()
            optimizer.zero_grad()
            output = model(inputs, meta, beta[question_id])
            # print("\nout\n")
            # print(output)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[question_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            
            loss = torch.sum((output - target) ** 2) + model.get_weight_norm()*lamb*0.5
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # optimizer.zero_grad()
            # target = output.clone()
            # #perform a dense refeeding step
            # output = model(output, meta, beta[question_id])
            # loss = torch.sum((output - target) ** 2) + model.get_weight_norm()*lamb*0.5
            # train_loss += loss.item()
            # optimizer.step()
        valid_acc = evaluate(model,beta, zero_train_data, valid_data, question_meta, question=question)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    plt.plot(epoch_array, train_array)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("Epoch vs Training accuracy")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, beta, train_data, valid_data, full_meta, question=False):
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
    for i, u in enumerate(valid_data["question_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        meta = full_meta[u].unsqueeze(0)
        output = model(inputs, meta,beta[u])

        guess = output[0][valid_data["user_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    question = True
    zero_train_matrix, train_matrix, valid_data, test_data, meta = load_data(base_path="../project/data", question=question)
    thetas, betas = get_thetas()
    eta = None
    if question:
        eta = betas
    else:
        eta = thetas
    model = AutoEncoder(train_matrix.shape[1], question=question)
    train(model, 0.1, 0.001, eta,  meta, train_matrix,  zero_train_matrix, valid_data, 300, question=question)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
