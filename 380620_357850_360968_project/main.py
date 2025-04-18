import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
import time
import matplotlib.pyplot as plt

np.random.seed(100)


def choose_hyperparameter_logistic_regression(data, test_data, labels, test_labels):
    nb_lr = 6
    nb_max_iter = 25
    learning_rates = np.logspace(-6, -1, num=nb_lr)
    max_iterations = np.linspace(100, 10_000, num=nb_max_iter).astype(int)
    print(f"Learning rates: {learning_rates}")
    print(f"Max Iterations: {max_iterations}")
    
    #creates 2 matrices with the values of the cross product of the lr and mi
    expanded_lr, expanded_mi = np.meshgrid(learning_rates, max_iterations)
    #we flatten the 2 matrices created above and combine them to finish the cross product
    coordinates_2d = np.column_stack([expanded_lr.ravel(), expanded_mi.ravel()])
    
    #third column to be filled with the loop results
    third_column = np.zeros((nb_max_iter * nb_lr, 1))
    
    #3D coordinates to plot and choose the best hyperparameters
    accuracy_results = np.hstack([coordinates_2d, third_column])
    macrof1_results = np.hstack([coordinates_2d, third_column])
    
    for i in range(nb_lr):
        for j in range(nb_max_iter):
            logistic_regression = LogisticRegression(learning_rates[i], max_iterations[j])
            logistic_regression.fit(data, labels)
            preds = logistic_regression.predict(test_data)
            acc = accuracy_fn(preds, test_labels)
            macrof1 = macrof1_fn(preds, test_labels)
            accuracy_results[i * nb_max_iter + j, 2] = acc
            macrof1_results[i * nb_max_iter + j, 2] = macrof1
            
    fig = plt.figure()
    
    log_learning_rates = np.log10(accuracy_results[:, 0])
    log_macrof1 = np.log10(macrof1_results[:, 0])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(log_learning_rates, accuracy_results[:, 1], accuracy_results[:, 2], c='blue', marker='o', label='Accuracy')
    ax.scatter(log_macrof1, macrof1_results[:, 1], macrof1_results[:, 2] * 100, c='red', marker='x', label='Macro F1')
    ax.set_xlabel("Learning rate (log10)")
    ax.set_ylabel("Max Iterations")
    ax.set_zlabel("Score (%)")
    ax.grid(True)
    
    plt.legend()
    plt.show()
    
    accuracy_extract = accuracy_results[:, 2]
    macrof1_extract = macrof1_results[:, 2]
    best_hyperparameters = np.lexsort((macrof1_extract, accuracy_extract))[-1]
    return np.hstack([accuracy_results[best_hyperparameters], macrof1_results[best_hyperparameters, 2]])


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load("../Data_MS1_2025/features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        fraction_train = 0.8 #New training set is 80% of the samples of the original training set, 20% remaining are validation set
        num_samples = xtrain.shape[0]
        rinds = np.random.permutation(num_samples)
        n_train = int(num_samples * fraction_train)
        xtest = xtrain[rinds[n_train:]] 
        ytest = ytrain[rinds[n_train:]] 

    #Normalizing data for logistic regression, k-means, knn
    if args.method == "logistic_regression" or args.method == "kmeans" or args.method == "knn":
        mean = np.mean(xtrain, axis=0)
        std = np.std(xtrain, axis=0)
        xtrain = normalize_fn(xtrain, mean, std)
        xtest = normalize_fn(xtest, mean, std)
    
    #Adding a bias term for logistic regression
    if args.method == "logistic_regression":
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(args.lr, args.max_iters)

    elif args.method == "kmeans":
        method_obj = KMeans(args.max_iters, args.centroids_init)
    
    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind="classification")

    

    s1 = time.time() #First time checkpoint

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    preds_train = method_obj.fit(xtrain, ytrain)
    
    s2 = time.time() #Second time checkpoint

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    
    s3 = time.time() #Third time checkpoint
    
    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"Training {args.method} took {s2 - s1} seconds")
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Predicting using {args.method} took {s3 - s2} seconds")
    print(f"{'Test' if not args.test else 'Validation'} set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    
    # if args.method == "logistic_regression":
    #     optimal_parameters = choose_hyperparameter_logistic_regression(xtrain, xtest, ytrain, ytest)
    #     print(f"The optimal values for the hyperparameters are attained for lr={optimal_parameters[0]} and max_iter={optimal_parameters[1]}")
    #     print(f"For the following optimal values: accuracy={optimal_parameters[2]} and macrof1={optimal_parameters[3]}")


if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="../Data_MS1_2025", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    parser.add_argument(
        "--centroids_init",
        default="random",
        type=str,
        help="how to initialize centers for kmeans",
    )


    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)