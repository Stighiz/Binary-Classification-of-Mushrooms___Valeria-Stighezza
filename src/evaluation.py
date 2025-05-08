import numpy as np
import pandas as pd
import models
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_plot_train_and_test(model, X_train, y_train, X_test, y_test, k_fold=None, model_type="tree"):
        """
        Performs predictions, evaluates the model, prints evaluation reports, and generates a combined
        subplot report for both training and test sets, including model parameters below the title.
        Parameters:
            model: Trained TreePredictor or RandomForest model with predict_and_evaluate method
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target labels.
            X_test (pd.DataFrame): Test feature data.
            y_test (pd.Series): Test target labels.
            k_fold (int, optional): Number of folds for cross-validation, if applicable.
            model_type (str): Type of model ('tree' or 'forest').
        Returns:
            tuple: (Evaluation, Evaluation) for train and test sets.
        """
        # Predictions and evaluation - train set
        train_predictions, train_correct, train_total, _ = model.predict_and_evaluate(X_train, y_train)
        train_evaluator = Evaluation(y_train.squeeze(), train_predictions,
                                    correct_predictions=train_correct, total_predictions=train_total)

        # Predictions and evaluation - test set
        test_predictions, test_correct, test_total, _ = model.predict_and_evaluate(X_test, y_test)
        test_evaluator = Evaluation(y_test.squeeze(), test_predictions,
                                    correct_predictions=test_correct, total_predictions=test_total)

        fig, (ax_train, ax_test) = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.4})
        fig.suptitle(f"Evaluation Report for {model_type} Model", fontsize=16, y=0.99)

        if model_type == "tree":
            params_text = f"Parameters: max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, criterion={model.criterion}"
            if k_fold is not None:
                params_text += f", n_folds={k_fold}"
        elif model_type == "forest":
            params_text = (f"Parameters: n_estimators={model.n_estimators}, max_depth={model.max_depth}, "
                        f"min_samples_split={model.min_samples_split}, criterion={model.criterion}, "
                        f"max_features={model.max_features}")
            if k_fold is not None:
                params_text += f", n_folds={k_fold}"
        else:
            raise ValueError("model_type must be 'tree' or 'forest'")

        fig.text(0.5, 0.95, params_text, ha='center', fontsize=12)

        # Plot confusion matrix - train set
        pad_size = 20
        sns.heatmap(train_evaluator.conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=train_evaluator.labels, yticklabels=train_evaluator.labels, ax=ax_train[0])
        ax_train[0].set_xlabel("Predicted Labels")
        ax_train[0].set_ylabel("True Labels")
        ax_train[0].set_title("Confusion Matrix (Train Set)", pad=pad_size)

        # Plot evaluation metrics - train set
        train_metrics = {
            "Accuracy": train_evaluator.accuracy(),
            "Precision": train_evaluator.precision(),
            "Recall": train_evaluator.recall(),
            "F1 Score": train_evaluator.f1_score(),
            "0-1 Loss": train_evaluator.zero_one_loss()
        }
        ax_train[1].bar(list(train_metrics.keys()), list(train_metrics.values()),
                        color=['mediumseagreen', 'seagreen', 'forestgreen', 'darkgreen', 'red'])
        ax_train[1].set_ylim(0, 1)
        ax_train[1].grid(True, axis='y', linestyle='--', alpha=0.8)
        for i, v in enumerate(train_metrics.values()):
            ax_train[1].text(i, v + 0.02, f"{v:.5f}", ha='center')
        ax_train[1].set_title("Evaluation Metrics (Train Set)", pad=pad_size)

        # Plot confusion matrix - test set
        sns.heatmap(test_evaluator.conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_evaluator.labels, yticklabels=test_evaluator.labels, ax=ax_test[0])
        ax_test[0].set_xlabel("Predicted Labels")
        ax_test[0].set_ylabel("True Labels")
        ax_test[0].set_title("Confusion Matrix (Test Set)", pad=pad_size)

        # Plot evaluation metrics - test set
        test_metrics = {
            "Accuracy": test_evaluator.accuracy(),
            "Precision": test_evaluator.precision(),
            "Recall": test_evaluator.recall(),
            "F1 Score": test_evaluator.f1_score(),
            "0-1 Loss": test_evaluator.zero_one_loss()
        }
        ax_test[1].bar(list(test_metrics.keys()), list(test_metrics.values()),
                    color=['mediumseagreen', 'seagreen', 'forestgreen', 'darkgreen', 'red'])
        ax_test[1].set_ylim(0, 1)
        ax_test[1].grid(True, axis='y', linestyle='--', alpha=0.8)
        for i, v in enumerate(test_metrics.values()):
            ax_test[1].text(i, v + 0.02, f"{v:.5f}", ha='center')
        ax_test[1].set_title("Evaluation Metrics (Test Set)", pad=pad_size)

        fig.text(0.5, -0.02,
                f"Train: {train_evaluator.correct_predictions} correct out of {train_evaluator.total_predictions} | "
                f"Test: {test_evaluator.correct_predictions} correct out of {test_evaluator.total_predictions}",
                ha='center', fontsize=12, wrap=True)

        ax_train[1].set_yticks(np.arange(0, 1.01, 0.05))
        ax_test[1].set_yticks(np.arange(0, 1.01, 0.05))
        plt.show()

        return train_evaluator, test_evaluator


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


class Evaluation:
    def __init__(self, y_true, y_pred, labels=None, correct_predictions=None, total_predictions=None):
        """
        Initializes the Evaluation object for computing and visualizing model performance metrics.
        Parameters:
            y_true (array-like): True class labels.
            y_pred (array-like): Predicted class labels.
            labels (array-like, optional): List of unique class labels. If None, computed from y_true.
            correct_predictions (int, optional): Number of correct predictions for reporting.
            total_predictions (int, optional): Total number of predictions for reporting.
        """
        self.y_true = np.array(y_true.squeeze())
        self.y_pred = np.array(y_pred)
        self.labels = np.unique(y_true) if labels is None else labels
        self.conf_matrix = self._compute_confusion_matrix()
        self.correct_predictions = correct_predictions
        self.total_predictions = total_predictions

    def _compute_confusion_matrix(self):
        """
        Computes the confusion matrix based on true and predicted labels.
        Returns:
            np.ndarray: Confusion matrix of shape (n_classes, n_classes).
        """
        n_classes = len(self.labels)
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        for true, pred in zip(self.y_true, self.y_pred):
            true_idx = label_to_index[true]
            pred_idx = label_to_index[pred]
            conf_matrix[true_idx, pred_idx] += 1

        return conf_matrix

    def accuracy(self):
        """
        Computes the accuracy of the model as the ratio of correct predictions to total predictions.
        Returns:
            float: Accuracy score.
        """
        correct = np.trace(self.conf_matrix)  # Sum of elements on the diagonal
        total = np.sum(self.conf_matrix)
        return correct / total

    def precision(self):
        """
        Computes the precision for each class and returns the average.
        Returns:
            float: Mean precision across all classes.
        """
        precisions = []
        for i in range(len(self.labels)):
            tp = self.conf_matrix[i, i]
            fp = np.sum(self.conf_matrix[:, i]) - tp
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return np.mean(precisions)

    def recall(self):
        """
        Computes the recall for each class and returns the average.
        Returns:
            float: Mean recall across all classes.
        """
        recalls = []
        for i in range(len(self.labels)):
            tp = self.conf_matrix[i, i]
            fn = np.sum(self.conf_matrix[i, :]) - tp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return np.mean(recalls)

    def f1_score(self):
        """
        Computes the F1-score as the harmonic mean of precision and recall.
        Returns:
            float: F1-score.
        """
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    def zero_one_loss(self):
        """
        Computes the 0-1 loss as the proportion of incorrect predictions.
        Returns:
            float: 0-1 loss value.
        """
        errors = np.sum(self.y_true != self.y_pred)
        return errors / len(self.y_true)

    def print_report(self, error_type, model_type):
        """
        Prints an evaluation report with all performance metrics.
        Parameters:
            criterion (str): The criterion used for the model (e.g., 'Entropy', 'Gini', 'MSE').
            error_type (str): Type of error, either 'train' or 'test'.
            model_type (str): Type of model ('tree' or 'forest').
        """
        if error_type not in ['Train', 'Test']:
            raise ValueError("error_type must be 'Train'or 'Test'")

        print(f"\n===== Evaluation Report for {model_type.capitalize()} model ({error_type} set) =====")

        print(f"Accuracy: {self.accuracy():.5f}")
        print(f"Precision: {self.precision():.5f}")
        print(f"Recall: {self.recall():.5f}")
        print(f"F1-score: {self.f1_score():.5f}")
        print(f"0-1 Loss: {self.zero_one_loss():.5f}")
        print("\nConfusion Matrix:")
        self._print_confusion_matrix()

        if self.correct_predictions is not None and self.total_predictions is not None:
            print(f"\nNumber of correct predictions: {self.correct_predictions} out of {self.total_predictions}")


    def _print_confusion_matrix(self):
        """
        Prints the confusion matrix in a readable format.
        """
        print(" ", " ".join(str(label) for label in self.labels))
        for i, row in enumerate(self.conf_matrix):
            print(self.labels[i], " ".join(f"{val:4}" for val in row))

    def plot_report(self, error_type, model_type, max_depth=None, min_samples_split=None, criterion=None, n_estimators=None, max_features=None):
        """
        Generates a graphical report with the confusion matrix and evaluation metrics.
        Parameters:
            error_type (str): Type of error, either 'train' or 'test'.
            model_type (str): Type of model ('tree' or 'forest').
        """
        if error_type not in ['Train', 'Test', 'Train CV', 'Test CV']:
            raise ValueError("error_type must be 'Train', 'Test', 'Train CV' or 'Test CV'")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"\nEvaluation Report for {model_type} model ({error_type} set)", fontsize=16, y=1.05)

        fig.text(0.5, 0.92,
                f"Parameters: max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}, n_estimators={n_estimators}, n_features={max_features}",
                ha='center', fontsize=12)

        # Plot confusion matrix
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels, ax=ax[0])
        ax[0].set_xlabel("Predicted Labels")
        ax[0].set_ylabel("True Labels")
        ax[0].set_title(f"Confusion Matrix\n")

        # Plot evaluation metrics
        metrics = {
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1 Score": self.f1_score(),
            "0-1 Loss": self.zero_one_loss()
        }
        ax[1].bar(list(metrics.keys()), list(metrics.values()), color=['mediumseagreen', 'seagreen', 'forestgreen', 'darkgreen', 'red'])
        ax[1].set_ylim(0, 1)
        ax[1].grid(True, axis='y', linestyle='--', alpha=0.8)  # Add grid
        for i, v in enumerate(metrics.values()):
            ax[1].text(i, v + 0.02, f"{v:.5f}", ha='center')
        ax[1].set_title(f"Evaluation Metrics\n")

        if self.correct_predictions is not None and self.total_predictions is not None:
            fig.text(0.5, -0.05, f"Number of correct predictions: {self.correct_predictions} out of {self.total_predictions}",
                     ha='center', fontsize=12, wrap=True)

        plt.tight_layout()
        plt.show()


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


def fit_and_evaluate(params):
    """
    Trains a decision tree and computes train/test accuracy and 0-1 loss for a given max_depth.
    Parameters:
        params: Tuple of (max_depth, min_samples_split, criterion, X_train, y_train, X_test, y_test)
    Returns:
        Tuple of (max_depth, train_accuracy, test_accuracy, train_loss, test_loss)
    """
    max_depth, min_samples_split, criterion, X_train, y_train, X_test, y_test = params
    print(f"Training tree with max_depth={max_depth}", flush=True)

    # Initialize and train the decision tree
    tree = models.TreePredictor(max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     criterion=criterion)
    tree.fit(X_train, y_train)

    # Predict on training and test sets
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    # Calculate accuracy (proportion of correct predictions)
    train_accuracy = np.mean(y_train.values.ravel() == y_pred_train)
    test_accuracy = np.mean(y_test.values.ravel() == y_pred_test)

    # Calculate 0-1 loss (proportion of incorrect predictions)
    train_loss = np.mean(y_train.values.ravel() != y_pred_train)
    test_loss = np.mean(y_test.values.ravel() != y_pred_test)

    print(f"Tree with max_depth={max_depth} completed", flush=True)
    return (max_depth, train_accuracy, test_accuracy, train_loss, test_loss)


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


def plot_performance_vs_depth(X_train, y_train, X_test, y_test, best_params, n_jobs: int = -1):
    """
    Plots train/test accuracy and 0-1 loss for varying max_depth values using parallel tree training.
    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test target values.
        best_params (dict): Dictionary containing best min_samples_split and criterion from tuning.
        n_jobs (int): Number of parallel jobs (-1 uses all available processors).
    """
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    max_depth_range = range(2, 51, 3)  # [2, 5, 8, ..., 47, 50]

    tasks = [
        (max_depth, best_params["min_samples_split"], best_params["criterion"],
         X_train, y_train, X_test, y_test)
        for max_depth in max_depth_range
    ]

    print(f"...building and evaluating {len(tasks)} trees using parallel computation...", flush=True)
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_evaluate)(task) for task in tasks
    )

    results.sort(key=lambda x: x[0])    # Sort results by max_depth
    max_depths, train_accuracies, test_accuracies, train_losses, test_losses = zip(*results)

    print("\nPerformance metrics for each max_depth:")
    for depth, train_acc, test_acc, train_loss, test_loss in results:
        print(f"max_depth={depth}: "
              f"Train Accuracy={train_acc:.5f}, Test Accuracy={test_acc:.5f}, "
              f"Train 0-1 Loss={train_loss:.5f}, Test 0-1 Loss={test_loss:.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Performance vs. Max Depth\nHyperparameters: min_samples_split={best_params['min_samples_split']}, criterion={best_params['criterion']}",
                 fontsize=14, fontweight='demibold')

    # Plot accuracy
    ax1.plot(max_depths, train_accuracies, label="Train Accuracy", color='b')
    ax1.plot(max_depths, test_accuracies, label="Test Accuracy", color='r')
    ax1.set_xticks(list(max_depth_range))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel("Max Depth", size=15)
    ax1.set_ylabel("Accuracy", size=15)
    ax1.set_title("Accuracy vs. Max Depth")
    ax1.legend(loc="upper left")

    # Plot 0-1 loss
    ax2.plot(max_depths, train_losses, label="Train 0-1 Loss", color='b')
    ax2.plot(max_depths, test_losses, label="Test 0-1 Loss", color='r')
    ax2.set_xticks(list(max_depth_range))
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel("Max Depth", size=15)
    ax2.set_ylabel("0-1 Loss", size=15)
    ax2.set_title("0-1 Loss vs. Max Depth")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()
