import numpy as np
import pandas as pd
import models
from joblib import Parallel, delayed

def kfold(n_samples, cv = 5):
    """
    Splits the data into training and test sets for cross-validation.
    Parameters:
        n_samples (int): Number of samples.
        cv (int): Number of folds (default=5).  
    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_indices, test_indices).
    """
    if cv < 2:
        raise ValueError('cv must be greater than 1')
    
    indices = np.random.permutation(n_samples)
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop

    return folds


###################################################################################


def evaluate_model(X_train, y_train, X_test, y_test, params, model_type):
    """
    Evaluates a model with given parameters and returns accuracy and 0-1 loss.
    Parameters:
        X_train, y_train: Training data and target.
        X_test, y_test: Test data and target.
        params: Hyperparameters for the model.
        model_type: str, "tree" for TreePredictor or "forest" for RandomForest.
    
    Returns:
        Dict[str, float]: Dictionary with accuracy and 0-1 loss on test set.
    """
    if model_type == "tree":
        model = models.TreePredictor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            criterion=params["criterion"]
        )
    elif model_type == "forest":
        model = models.RandomForest(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            criterion=params["criterion"],
            max_features=params["max_features"]
        )
    else:
        raise ValueError("Model_type must be 'tree' or 'forest'")
    
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_test_arr = y_test.values.ravel()
    return {
        "accuracy": np.mean(y_test_arr == y_pred_test),
        "loss": np.mean(y_test_arr != y_pred_test)
    }


###################################################################################


def perform_grid_search(X_train, y_train, X_test, y_test, cv = None, n_jobs = -1, model_type = "tree", param_grid = None):
    """
    Performs grid search with optional cross-validation to find the best hyperparameters.
    Parameters:
        X_train (pd.DataFrame): Training dataset.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame, optional): Test dataset (required if cv=None).
        y_test (pd.Series, optional): Test target (required if cv=None).
        cv (int, optional): Number of CV folds. If None, uses X_test/y_test.
        n_jobs (int): Number of parallel jobs (-1 means all processors).
        model_type (str): "tree" for TreePredictor or "forest" for RandomForest.
        param_grid (List[Dict], optional): Custom parameter grid. If None, uses default.
    Returns:
        grid_search_df (pd.DataFrame): Results of all combinations.
        best_params (pd.Series): Best hyperparameters and performance.
    """
    if param_grid is None:
        if model_type == "tree":
            param_grid = [
                {"max_depth": md, "min_samples_split": mss, "criterion": c}
                for md in [10, 25, 50]  
                for mss in [2, 10, 50]
                for c in ['gini', 'entropy', 'MSE']
            ]
        elif model_type == "forest":
            param_grid = [
                {"n_estimators": ne, "max_depth": md, "min_samples_split": mss, "criterion": c, "max_features": mf}
                for ne in [10, 20]          
                for md in [30, 50]      
                for mss in [2, 10]     
                for c in ['gini', 'entropy', 'MSE']
                for mf in ["sqrt"]
            ]
        else:
            raise ValueError("model_type must be 'tree' or 'forest'")

    y_train_arr = y_train.values.ravel()

    
    def evaluate_params(params):
        """
        Evaluates a single hyperparameter configuration for a decision tree or random forest model.
        Based on the validation mode:
        -   If `cv` is None: trains the model on the provided `X_train` and `y_train`,
            and evaluates it on `X_test` and `y_test`.
        -   If `cv` is an integer: performs k-fold cross-validation using the `kfold()` function.
        Parameters:
            params (Dict[str, Any]): Dictionary containing a specific set of hyperparameters to evaluate.
        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameter values and the evaluation metrics:
                - If `cv` is None: includes train/test accuracy and loss.
                - If `cv` is provided: includes cross-validated test accuracy and loss only
                (under keys "accuracy_test" and "loss_test").
        """
        if cv is None:
            if X_test is None or y_test is None:
                raise ValueError("X_test and y_test must be provided")
            y_test_arr = y_test.values.ravel()
            if model_type == "tree":
                model = models.TreePredictor(**params)
            else:
                model = models.RandomForest(**params, random_state=42)
            

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            return {
                **params,
                "accuracy_train": np.mean(y_train_arr == y_pred_train),
                "accuracy_test": np.mean(y_test_arr == y_pred_test),
                "loss_train": np.mean(y_train_arr != y_pred_train),
                "loss_test": np.mean(y_test_arr != y_pred_test),
                "model": model,
            }
        else:
            folds = kfold(len(X_train), cv)
            cv_scores = []
            cv_losses = []
            for train_idx, test_idx in folds:
                X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
                y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
                metrics = evaluate_model(X_tr, y_tr, X_te, y_te, params, model_type)
                cv_scores.append(metrics["accuracy"])
                cv_losses.append(metrics["loss"])
            mean_cv_score = np.mean(cv_scores)
            mean_cv_loss = np.mean(cv_losses)
            return {
                **params,  
                "accuracy_train": None,  # not computed with CV
                "accuracy_test": mean_cv_score,  # CV mean accuracy
                "loss_test": mean_cv_loss  # CV mean 0-1 loss
            }
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) for params in param_grid       # run grid search in parallel
    )

    grid_search_df = pd.DataFrame(results)
    best_params = grid_search_df.sort_values("accuracy_test", ascending=False).iloc[0]
    
    # Print results
    print(f"Best parameters for {model_type}:")
    for key, value in best_params.items():
        if key not in ["accuracy_train", "accuracy_test", "loss_train", "loss_test"] and key != "model":
            print(f"  {key}: {value}")
    if cv is None:
        print(f"  Train accuracy: {best_params['accuracy_train']:.4f}")
        print(f"  Train loss: {best_params['loss_train']:.4f}")
        print(f"  Test accuracy: {best_params['accuracy_test']:.4f}")
        print(f"  Test loss: {best_params['loss_test']:.4f}")
    else:
        print(f"  CV mean Test accuracy: {best_params['accuracy_test']:.4f}")
        print(f"  CV mean Test loss: {best_params['loss_test']:.4f}\n")

    if cv is None:
        return best_params
    
    else:   # CV tree case
        # Train a tree with the best parameters on the full training set
        best_tree = models.TreePredictor(max_depth=best_params.max_depth, criterion=best_params.criterion, min_samples_split=best_params.min_samples_split)
        best_tree.fit(X_train, y_train)
        return best_params, best_tree
        
   


    



