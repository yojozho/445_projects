"""
EECS 445 - Winter 2024

Project 1 main file.
"""


import itertools
import string
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC

import helper

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import seaborn as sns

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)




def extract_word(input_string: str) -> list[str]:
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along
    whitespace. Return the resulting array.

    Example:
        > extract_word("I love EECS 445. It's my favorite course!")
        > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Args:
        input_string: text for a single review

    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    input_string = input_string.lower()

    for i in input_string:
        if i in string.punctuation:
            input_string = input_string.replace(i, " ")

    input_list = input_string.split(" ")
    input_list = [i for i in input_list if i != '']

    return input_list


def extract_dictionary(df: pd.DataFrame) -> dict[str, int]:
    """
    Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words mapping from each
    distinct word to its index (ordered by when it was found).

    Example:
        Input df:

        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

        The output should be a dictionary of indices ordered by first occurence in
        the entire dataset. The index should be autoincrementing, starting at 0:

        {
            it: 0,
            was: 1,
            the: 2,
            best: 3,
            of: 4,
            times: 5,
            blurst: 6,
        }

    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    k = 0

    for i in df.reviewText:
        processed_setence = extract_word(i)
        for j in processed_setence:
            if j not in word_dict:
                word_dict.update({j: k})
                k += 1

    return word_dict


def generate_feature_matrix(
    df: pd.DataFrame, word_dict: dict[str, int]
) -> npt.NDArray[np.float64]:
    """
    Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Args:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices

    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))

    k = 0

    for i in df.reviewText: 
        processed_setence = extract_word(i)
        for j in processed_setence:
            if j in word_dict:
                feature_matrix[k, word_dict[j]] = 1
        k += 1

    return feature_matrix




def performance(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
    metric: str = "accuracy",
) -> np.float64:
    """
    Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Args:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')

    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    if metric == 'accuracy': 
        return (tp + tn) / (tp + tn + fp + fn)
    elif metric == 'f1-score':
        return (2 * tp) / (2 * tp + fp + fn)
    elif metric == 'precision':
        return tp / (tp + fp)
    elif metric == 'sensitivity':
        return tp / (tp + fn)
    elif metric == 'specificity':
        return tn / (tn + fp)


def cv_performance(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
) -> float:
    """
    Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.

    Args:
        clf: an instance of LinearSVC() or SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1, -1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')

    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    # Put the performance of the model on each fold in the scores array
    scores = []
    folds = StratifiedKFold(n_splits = k)
    for train, test in folds.split(X, y):
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            if metric == 'auroc':
                y_pred = clf.decision_function(X_test)
            else:
                y_pred = clf.predict(X_test)
            scores.append(performance(y_test, y_pred, metric))

    return np.array(scores).mean()


def select_param_linear(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    C_range: list[float] = [],
    loss: str = "hinge",
    penalty: str = "l2",
    dual: bool = True,
) -> float:
    """
    Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")

    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    bestC = 0
    bestPerformance = 0
    for c in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = dual, C = c, random_state = 445)
        performance = cv_performance(clf, X, y, k, metric)
        if performance > bestPerformance:
            bestC = c
            bestPerformance = performance

    return bestC


def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    penalty: str,
    C_range: list[float],
    loss: str,
    dual: bool,
) -> None:
    """
    Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor

    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = dual, C = c, random_state = 445)
        clf.fit(X, y)
        norm0.append(np.count_nonzero(clf.coef_))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    param_range: npt.NDArray[np.float64] = [],
) -> tuple[float, float]:
    """
    Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.

    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    bestPerformance = 0
    
    for c, r, in param_range:
        clf = SVC(kernel = 'poly', degree = 2, C = c, coef0 = r, gamma = 'auto', random_state = 445)
        performance = cv_performance(clf, X, y, k, metric)
        print("CV performance for C =", c, "and r =", r, ":", performance)
        if performance > bestPerformance:
            best_C_val = c
            best_r_val = r
            bestPerformance = performance

    return best_C_val, best_r_val


def train_word2vec(filename: str) -> Word2Vec:
    """
    Train a Word2Vec model using the Gensim library.

    First, iterate through all reviews in the dataframe, run your extract_word() function
    on each review, and append the result to the sentences list. Next, instantiate an
    instance of the Word2Vec class, using your sentences list as a parameter and using workers=1.

    Args:
        filename: name of the dataset csv

    Returns:
        created Word2Vec model
    """
    df = helper.load_data(filename)
    sentences = []
    # TODO: Complete this function
    for i in df.reviewText:
        processed_sentence = extract_word(i)
        sentences.append(processed_sentence)
    word_vec = Word2Vec(sentences, workers = 1)
    return word_vec


def compute_association(filename: str, w: str, A: list[str], B: list[str]) -> float:
    """
    Args:
        filename: name of the dataset csv
        w: a word represented as a string
        A: set of English words
        B: set of English words

    Returns:
        association between w, A, and B as defined in the spec
    """
    model = train_word2vec(filename)

    # First, we need to find a numerical representation for the English language words in A and B

    # TODO: Complete words_to_array()
    def words_to_array(s: list[str]) -> npt.NDArray[np.float64]:
        """Convert a list of string words into a 2D numpy array of word embeddings,
        where the ith row is the embedding vector for the ith word in the input set (0-indexed).

            Args:
                s (list[str]): List of words to convert to word embeddings

            Returns:
                npt.NDArray[np.float64]: Numpy array of word embeddings
        """
        word_embeddings = np.zeros((len(s), model.vector_size))
        for i in range(len(s)):
            word_embeddings[i] = model.wv[s[i]]
        return word_embeddings

    # TODO: Complete cosine_similarity()
    def cosine_similarity(
        array: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate the cosine similarities between w and the input set.

        Args:
            array: array representation of the input set
            w: word embedding for w

        Returns:
            1D Numpy Array where the ith element is the cosine similarity between the word
            embedding for w and the ith embedding in input set
        """
        similarity_array = []
        for input in array:
            w_mag = np.linalg.norm(w, axis = 0, keepdims = True)
            input_mag = np.linalg.norm(input, axis = 0, keepdims = True)
            cosine = np.dot(w, input)/np.matmul(w_mag, input_mag)
            similarity_array.append(cosine)
        similarity_array = np.array(similarity_array)
        return similarity_array

    # Although there may be some randomness in the word embeddings, we have provided the
    # following test case to help you debug your cosine_similarity() function:
    # This is not an exhaustive test case, so add more of your own!
    test_arr = np.array([[4, 5, 6], [9, 8, 7]])
    test_w = np.array([1, 2, 3])
    test_sol = np.array([0.97463185, 0.88265899])
    assert np.allclose(
        cosine_similarity(test_arr, test_w), test_sol, atol=0.00000001
    ), "Cosine similarity test 1 failed"

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A,
    #      and the mean cosine similarity between w and the words in B
    A = words_to_array(A)
    B = words_to_array(B)
    w = model.wv[w]

    A_scores = cosine_similarity(A, w)
    B_scores = cosine_similarity(B, w)

    s = A_scores.mean() - B_scores.mean()

    return s

def preprocessData(df: pd.DataFrame) -> dict[str, int]:
    """ Preprocess text data. """
    word_dict = {}
    nltk_stopwords = set(stopwords.words('english'))
    k = 0
    for sentence in df.reviewText:
        processed = extract_word(sentence)
        processed = [word for word in processed if word not in nltk_stopwords]
        processed = [word for word in processed if len(word) != 1]
        processed = [word for word in processed if word.isnumeric() is False]
        for j in processed:
            if j not in word_dict:
                word_dict.update({j: k})
                k += 1

    return word_dict


def performance_multi(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
) -> np.float64:
    """ Calculate performance score. """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels = [-1, 0, 1])
    tp = 0
    n = len(y_pred)
    for i in range(len(confusion_matrix)):
        tp += confusion_matrix[i][i]
    return tp / n



def cv_performance_multi(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
) -> float:
    """ Run cross-validation. """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    # Put the performance of the model on each fold in the scores array
    scores = []
    folds = StratifiedKFold(n_splits = k)
    for train, test in folds.split(X, y):
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            if metric == 'auroc':
                y_pred = clf.decision_function(X_test)
            else:
                y_pred = clf.predict(X_test)
            scores.append(performance_multi(y_test, y_pred))

    return np.array(scores).mean()


def select_param(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    C_range: list[float] = [],
    loss: str = "hinge",
    penalty: str = "l2",
    dual: bool = True,
) -> float:
    """ Select the best C hyper parameter. """
    bestC = 0
    bestPerformance = 0
    for c in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = dual, C = c, random_state = 445)
        performance = cv_performance_multi(clf, X, y, k, metric)
        if performance > bestPerformance:
            bestC = c
            bestPerformance = performance

    return bestC

def main() -> None:
    # Read binary data
    # NOTE: Use the X_train, Y_train, X_test, and Y_test provided below as the training set and test set
    #       for the reviews in the file you read in.
    #
    #       Your implementations of extract_dictionary() and generate_feature_matrix() will be called
    #       to produce these training and test sets (for more information, see get_split_binary_data() in helper.py).
    #       DO NOT reimplement or edit the code we provided in get_split_binary_data().
    #
    #       Please note that dictionary_binary will not be correct until you have correctly implemented extract_dictionary(),
    #       and X_train, Y_train, X_test, and Y_test will not be correct until you have correctly
    #       implemented extract_dictionary() and generate_feature_matrix().
    filename = "data/dataset.csv"


    X_train, Y_train, X_test, Y_test, dictionary_binary = helper.get_split_binary_data(
        filename=filename
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = helper.get_imbalanced_data(
        dictionary_binary, filename=filename
    )

    # TODO: Questions 2, 3, 4, 5
    #!!!The implementations are in the writeup sections of project!!! 
    #I wrote everything in a jupyter notebook so I could display the outputs easier#

    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = helper.get_multiclass_training_data()

    heldout_features = helper.get_heldout_reviews(multiclass_dictionary)


if __name__ == "__main__":
    main()
