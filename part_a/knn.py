from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import *
import numpy


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.
    
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    #print(mat.shape)
    #print(mat)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc, mat


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # TASK 1.b)
    # Given Question A and Question B, the underlying assumption is that,
    # if question A and Question B is scored the same by other students, 
    # then question A and Question B have the same correctness for certain students.

    nbrs = KNNImputer(n_neighbors=k)
    matrix = numpy.rot90(matrix) # rotates sparse matrix. So, questions are now samples and students are features.

    mat = nbrs.fit_transform(matrix)
    mat = numpy.rot90(mat) # rotates fitted matrix back for eval

    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy for questions: {}".format(acc))
    return acc, mat


def main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)



    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # TASK 1.a)
    k_lst = [1, 6, 11, 16, 21, 26]
    user_results = []
    stored_mat_user = []
    for k in k_lst:
        final = knn_impute_by_user(sparse_matrix, val_data, k)
        user_results.append(final[0])
        stored_mat_user.append(final[1])
    best_k = user_results.index(max(user_results))
    print("Test Acc with best k for students: " + str(sparse_matrix_evaluate(test_data, stored_mat_user[best_k])), k_lst[best_k])

    
    plt.plot(k_lst, user_results, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation ')
    plt.title('Number of Neighbors vs Validation')
    plt.grid(True)
    plt.show()


    # TASK 1.c)
    k_lst = [1, 6, 11, 16, 21, 26]
    results = []
    stored_mat_item = []
    for k in k_lst:
        final = knn_impute_by_item(sparse_matrix, val_data, k)
        results.append(final[0])
        stored_mat_item.append(final[1])
    best_k = results.index(max(results))
    print("Test Acc with best k for items: " + str(sparse_matrix_evaluate(test_data, stored_mat_item[best_k])), k_lst[best_k])


    plt.plot(k_lst, results, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation ')
    plt.title('Number of Neighbors vs Validation')
    plt.grid(True)
    plt.show()


    # TASK 1.e)
    # - KNN performs poorly with spare data. We see that the given training set has many values that need to be filled in.
    #   This means there is alot of empty space and limited overlap between data points, so the closest nodes we use for our prediction
    #   may be very far away from our target. In example, for user based filtering, given we are trying to predict an answer for a student,
    #   if there are no students with similar diagnostic answers, then the nearest neighbours probably don't represent the local structure of the data.
    
    # - Slow. We will have to calculate the distance for between each and every datapoint, for every empty index in our sparse matrix. 
    #   Since we want to scale our application to get data from more students and increase the number of questions students can do, 
    #   the large data set will incur a high computational cost. 

    pass


if __name__ == "__main__":
    main()
