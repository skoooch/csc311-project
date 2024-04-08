from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x): # I think my derivaton for gradient descent is correct, but I am underflowing here. 
    """ Apply sigmoid function.
    """
    return (np.exp(x)) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO: COMPLETE                                                           
    # Implement the function as described in the docstring.             #
    #####################################################################\
    #print((theta[data["user_id"]] - beta[data["question_id"]]), "DAWFWAFAWGWAGAW") # temp returning 0 since e^x is underflows to 0.
    #temp = sigmoid((theta[data["user_id"]] - beta[data["question_id"]]))

    #print("temp", np.sum(temp == 0))
    #print(temp, "temp array")
    # log_lklihood = np.sum(data["is_correct"]*np.log(temp) + (1 - data["is_correct"])*np.log(1 - temp))
    log_lklihood = np.sum(data["is_correct"]*(theta[data["user_id"]] - beta[data["question_id"]]) - np.log(1+np.exp( (theta[data["user_id"]] - beta[data["question_id"]]) )))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    # what?
    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float (loss rate)
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO: COMPLETE                                                     #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # theta calculations

    for i in range(len(theta)):
        irows = np.where(data["user_id"] == i)[0]
        theta_deriv = np.sum(data["is_correct"][irows] - sigmoid(theta[i] - beta[data["question_id"][irows]])) *-1
        theta[i] -= lr * theta_deriv

    # beta calculations

    for j in range(len(beta)):
        jrows = np.where(data["question_id"] == j)[0]
        beta_deriv = np.sum(sigmoid(theta[data["user_id"][jrows]] - beta[j]) - data["is_correct"][jrows]) *-1
        beta[j] -= lr * beta_deriv

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    """temp = sigmoid(theta[data["user_id"]] - beta[data["question_id"]])
    theta_deriv = np.sum((data["is_correct"])*(1 - temp) - (1 - data["is_correct"])*(temp))
    theta = (theta - (lr * theta_deriv))"""
    """temp = sigmoid(theta[data["user_id"]] - beta[data["question_id"]])
    beta_deriv = np.sum((1 - data["is_correct"])*(temp) - (data["is_correct"])*(1 - temp))
    beta = (beta - (lr *beta_deriv))"""
    return theta, beta


def irt(data, val_data, lr, iterations): 
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    data["user_id"] = np.array(data["user_id"])
    data["question_id"] = np.array(data["question_id"])
    data["is_correct"] = np.array(data["is_correct"])
    theta = np.zeros(len(np.unique(data["user_id"]))) 
    beta = np.zeros(len(np.unique(data["question_id"])))
    val_acc_lst = []
    likelihoods = []
    val_likelihoods = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        likelihoods.append(neg_lld)
        val_likelihoods.append(val_neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, likelihoods, val_likelihoods 


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])



def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    #print(np.unique(train_data["is_correct"]))
    #print(np.unique(train_data["user_id"]))
    #print(np.unique(train_data["question_id"]))
    #print(sparse_matrix)
    
    print([train_data["question_id"][0]])
    print([train_data["question_id"][1]])
    print([train_data["question_id"][2]])

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    training_ll = []
    val_ll = []
    val_accs = []
    lrs = [0.1, 0.001, 0.0001, 0.00001]
    iters = [100, 200, 500, 1000, 1500, 2000]
    #lrs = [0.01] FOR TESTING
    #iters = [20] FOR TESTING
    tuples = []
    thetas = []
    betas = []
    for l in lrs:
        for iter in iters:
            result = irt(train_data, val_data, l, iter)
            val_accs.append(result[2][-1])
            training_ll.append(result[3])
            val_ll.append(result[4])
            print(training_ll)
            print(val_ll)
            tuples.append((l, iter))
            thetas.append(result[0])
            betas.append(result[1])

    # model with highest accuracy
    best_index = val_accs.index(max(val_accs))


    iterations = range(0, len(val_ll[best_index]))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, training_ll[best_index], label='Training Log Likelihood')
    plt.plot(iterations, val_ll[best_index], label='Validation Log Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Training and Validation Log Likelihoods vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("learning rate and iterations", tuples[best_index])
    print("best validation acc", val_accs[best_index])
    print("test acc", evaluate(test_data, thetas[best_index], betas[best_index]))
    # best result is 0.0001 LR, 1500 iterations.
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    score_one = []
    score_two = []
    score_three = []
    for i in range(len(thetas[best_index])):
        jone = {"user_id":np.array([i]), "question_id": np.array([train_data["question_id"][0]]), "is_correct": np.array([1])}
        jtwo = {"user_id":np.array([i]), "question_id": np.array([train_data["question_id"][1]]), "is_correct": np.array([1])}
        jthree = {"user_id":np.array([i]), "question_id": np.array([train_data["question_id"][2]]), "is_correct": np.array([1])}
        score_one.append(sigmoid(thetas[best_index][i] - betas[best_index][train_data["question_id"][0]]))
        score_two.append(sigmoid(thetas[best_index][i] - betas[best_index][train_data["question_id"][1]]))
        score_three.append(sigmoid(thetas[best_index][i] - betas[best_index][train_data["question_id"][2]]))
    
    # line plot
    print(score_one, score_two, score_three) 
    plt.figure(figsize=(10, 6))
    plt.plot(thetas[best_index], score_one, label='Question j1')
    plt.plot(thetas[best_index], score_two, label='Question j2')
    plt.plot(thetas[best_index], score_three, label='Question j3')
    plt.xlabel('Ability (Theta)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Ability for Different Questions')
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(thetas[best_index], score_one, label='Question j1', marker='o', s=100)
    plt.scatter(thetas[best_index], score_two, label='Question j2', marker='s', s=100)
    plt.scatter(thetas[best_index], score_three, label='Question j3', marker='^', s=100)
    plt.xlabel('Ability (Theta)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Ability for Different Questions')
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
