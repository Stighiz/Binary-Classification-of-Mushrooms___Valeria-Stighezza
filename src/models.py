import numpy as np
import pandas as pd
import sys
from joblib import Parallel, delayed
import graphviz 

class Node:

    def __init__(self, feature_name=None, feature_index=None, threshold=None, left=None, right=None, leaf_value=None, categorical=False):
        """
        Initializes a decision tree node, representing either a split or a leaf.
        Parameters:
            feature_name (str, optional): Name of the feature used for splitting.
            feature_index (int, optional): Index of the feature used for splitting. 
            threshold (float or str, optional): Threshold value for the split (numerical) or category (categorical). 
            left (Node, optional): Left child subtree. 
            right (Node, optional): Right child subtree. 
            leaf_value (any, optional): Predicted value (only if the node is a leaf). 
            categorical (bool, optional): True if the feature is categorical. 
        Attributes:
            most_freq_class (any): Most frequent class reaching the node, set later. 
            info_gain (float): Information gain of the split, set later. 
            criterion (str): Splitting criterion ("entropy", "gini", "MSE"), set later. 
        """
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.categorical = categorical
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value = leaf_value
        self.most_freq_class = None
        self.info_gain = None
        self.criterion = None
                        

    def is_leaf(self):
        """
        Checks if the node is a leaf.
        Returns:
            bool: True if the node is a leaf (has a predicted value), False otherwise.
        """
        return self.leaf_value is not None


    def set_split_info(self, info_gain=None, feature_name=None, most_freq_class=None, criterion=None):
        """
        Sets metadata for a split node.
        Parameters:
            info_gain (float, optional): Information gain achieved by the split. 
            feature_name (str, optional): Name of the feature used for splitting. 
            most_freq_class (any, optional): Most frequent class among samples reaching the node. 
            criterion (str, optional): Splitting criterion ("entropy", "gini", "MSE"). 
        """
        self.info_gain = info_gain
        self.feature_name = feature_name
        self.most_freq_class = most_freq_class
        self.criterion = criterion


    def __str__(self):
        """
        Returns a string representation of the node.
        Returns:
            str: Description of the node, including feature, threshold, and split information (for split nodes) or predicted class (for leaf nodes).
        """
        if self.is_leaf():
            return f"Leaf Node with Class: {self.leaf_value}"
        op = "=" if self.categorical else "<="
        thr_str = f"{self.threshold:.3f}" if not self.categorical else str(self.threshold)
        return (f"Node splits on feature: {self.feature_name} {op} {thr_str}\n"
                f"Information Gain: {self.info_gain:.4f}\n"
                f"Majority class: {self.most_freq_class}\n\n")


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


class TreePredictor:

    def __init__(self, min_samples_split=10, max_depth=15, criterion="entropy"):
        """
        Initializes the TreePredictor for building a decision tree classifier.
        Parameters:
            min_samples_split (int, optional): Minimum number of samples required to split a node. Default is 2.
            max_depth (int, optional): Maximum depth of the decision tree. Default is 100.
            criterion (str, optional): Criterion for evaluating splits ("entropy", "gini", "MSE"). Default is "entropy".
        Attributes:
            n_classes (int): Number of unique classes in the target variable, set during training. Initialized as None.
            root (Node): Root node of the decision tree, set during training. Initialized as None.
            feature_names (list): Names of input features, set during training. Initialized as None.
            n_features (int): Number of input features, set during training. Initialized as None.
            feature_importance_ (np.ndarray): Importance scores for each feature, set during training. Initialized as None.
            feature_types (dict): Dictionary mapping feature names to their data types, set during training. Initialized as None.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_classes = None
        self.root = None
        self.feature_names = None
        self.n_features = None
        self.feature_importance_ = None
        self.feature_types = None


    def fit(self, X, y) -> None:
        """
        Trains the decision tree classifier on the provided training data.
        Parameters:
            X (pd.DataFrame): DataFrame containing the feature data for training.
            y (pd.Series or pd.DataFrame): Series or single-column DataFrame containing the target labels.
        """
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame.")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a Series.")

        self.feature_names = X.columns
        self.n_features = X.shape[1]
        self.feature_importance_ = np.zeros(self.n_features)
        self.feature_types = {col: X[col].dtype for col in X.columns}

        print(f'Starting training of the decision tree using \n\t- Splitting criterion = {self.criterion}\n\t- Maximum depth = {self.max_depth}\n\t- Minimum number of samples required in a node for splitting = {self.min_samples_split} ...')
        # Recursively constructs the decision tree
        self.root = self._grow_tree(X.values, y.values, depth=0)
        print("Tree successfully fitted!\n")


    def _is_categorical_feature(self, feat_idx):
        """
        Determines if a feature is categorical based on its index.
        Parameters:
            feat_idx (int): Index of the feature to check.
        Returns:
            bool: True if the feature is categorical (dtype 'object'), False otherwise.
        """
        dtype = self.feature_types[self.feature_names[feat_idx]]
        return dtype == 'object' 


    def _grow_tree(self, X, y, depth=0, verbose=False):
        """
        Recursively constructs the decision tree by splitting nodes or creating leaves.
        Parameters:
            X (np.ndarray): NumPy array of feature data (shape: n_samples, n_features).
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
            depth (int, optional): Current depth of the tree, starting at 0 for the root. 
            verbose (bool, optional): If True, prints debugging information during construction. 
        Returns:
            Node: A Node object representing either a leaf or an internal split node.
        """

        if verbose:
            sys.stdout.flush()
            print("Inside _grow_tree, depth =", depth)

        n_samples, _ = X.shape
        unique_labels = np.unique(y)

        # Stopping conditions: maximum depth reached, insufficient samples, or homogeneous labels
        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(unique_labels) == 1):
            leaf_val = self._most_common_label(y)
            if verbose:
                print(f"Creating leaf node at depth {depth} with class: {leaf_val}")
            return Node(leaf_value=leaf_val)

       # Finds the best split for the current data
        best_feat, best_thresh, best_gain, left_idxs, right_idxs = self._find_best_split(X, y)

        # If no valid split is found, returns a leaf node
        if best_feat is None or len(right_idxs) == 0 or len(left_idxs) == 0:
            leaf_val = self._most_common_label(y)
            if verbose:
                print(f"No valid split at depth {depth}, creating leaf with class: {leaf_val}")
            return Node(leaf_value=leaf_val)

        self.feature_importance_[best_feat] += best_gain

        # Creates a new node with the best split information
        node = Node(feature_name=self.feature_names[best_feat],
                    feature_index=best_feat,
                    threshold=best_thresh,
                    categorical=self._is_categorical_feature(best_feat))
        node.set_split_info(info_gain=best_gain,
                            feature_name=self.feature_names[best_feat],
                            most_freq_class=self._most_common_label(y),
                            criterion=self.criterion)

        if verbose:
            print(f"Splitting on feature {self.feature_names[best_feat]} with threshold {best_thresh} at depth {depth}")

        # Recursively constructs the left and right subtrees
        left_subtree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth=depth+1)
        right_subtree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth=depth+1)
        node.left = left_subtree
        node.right = right_subtree
        return node


    def _find_best_split(self, X, y):
        """
        Identifies the optimal feature and threshold for splitting the current node.
        Parameters:
            X (np.ndarray): NumPy array of feature data (shape: n_samples, n_features).
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
        Returns:
            best_feat (int or None): Index of the best feature to split on, or None if no valid split is found.
            best_thresh (float or str or None): Optimal threshold (numerical or categorical), or None if no valid split.
            best_gain (float): Information gain achieved by the best split, or -inf if no valid split.
            best_left_idxs (np.ndarray or None): Indices of samples in the left branch, or None if no valid split.
            best_right_idxs (np.ndarray or None): Indices of samples in the right branch, or None if no valid split.
        """
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        best_left_idxs, best_right_idxs = None, None
        current_impurity = self._calc_impurity(y)

        for feat_idx in range(self.n_features):
            X_column = X[:, feat_idx]
            values = np.unique(X_column)
            for thresh in values:
                left_idxs, right_idxs = self._split_data(X_column, thresh, feat_idx)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                gain = self._calc_info_gain(y, left_idxs, right_idxs, current_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
                    best_left_idxs, best_right_idxs = left_idxs, right_idxs

        return best_feat, best_thresh, best_gain, best_left_idxs, best_right_idxs


    def _calc_info_gain(self, y, left_idxs, right_idxs, current_impurity):
        """
        Calculates the information gain for a proposed split.
        Parameters:
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
            left_idxs (np.ndarray): Indices of samples in the left branch.
            right_idxs (np.ndarray): Indices of samples in the right branch.
            current_impurity (float): Impurity of the current node before splitting.
        Returns:
            float: Information gain achieved by the split, or 0 if the split is invalid.
        """
        n = len(y)
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0

        impurity_left = self._calc_impurity(y[left_idxs])
        impurity_right = self._calc_impurity(y[right_idxs])
        weighted_impurity = (n_left/n) * impurity_left + (n_right/n) * impurity_right
        info_gain = current_impurity - weighted_impurity
        return info_gain


    def _calc_impurity(self, y):
        """
        Calculates the impurity of a node based on the specified criterion.
        Parameters:
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
        Returns:
            float: Impurity value based on the criterion ("scaled_entropy", "gini", or "MSE").
        """
        if y.dtype == "O":
            y = pd.factorize(y)[0]  # Convert categorical labels to numerical if necessary

        if self.criterion == "gini":
            return self._gini_impurity(y)
        elif self.criterion == "entropy":
            return self._scaled_entropy(y)
        elif self.criterion == "MSE":
            return self._squared_impurity(y)
        else:
            raise ValueError("Unsupported criterion.")


    def _gini_impurity(self, y):
        hist = np.bincount(y, minlength=2)
        ps = hist / len(y)
        return 2 * ps[0] * (1 - ps[0])


    def _scaled_entropy(self, y):
        hist = np.bincount(y, minlength=2)
        ps = hist / len(y)
        p = ps[0]
        if p == 0 or p == 1:
            return 0
        return -(p/2) * np.log2(p) - ((1 - p)/2) * np.log2(1 - p)


    def _squared_impurity(self, y):
        hist = np.bincount(y, minlength=2)
        ps = hist / len(y)
        return np.sqrt(ps[0] * (1 - ps[0]))


    def _split_data(self, X_column, threshold, feat_idx):
        """
        Splits the data into left and right branches based on a feature’s threshold.
        Parameters:
            X_column (np.ndarray): Values of the feature to split (shape: n_samples).
            threshold (float or str): Threshold value (numerical or categorical) for the split.
            feat_idx (int): Index of the feature being split.
        Returns:
            left_idxs (np.ndarray): Indices of samples in the left branch.
            right_idxs (np.ndarray): Indices of samples in the right branch.
        """
        if self._is_categorical_feature(feat_idx):
            left_idxs = np.argwhere(X_column == threshold).flatten()
            right_idxs = np.argwhere(X_column != threshold).flatten()
        else:
            left_idxs = np.argwhere(X_column <= threshold).flatten()
            right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs


    def _most_common_label(self, y):
        """
        Identifies the most frequent label in a set of target values.
        Parameters:
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
        Returns:
            any: The most common label in the input array.
        """
        freq = {}
        for label in y:
            if label in freq: freq[label] += 1
            else: freq[label] = 1

        most_common_label = None
        max_count = 0
        for label, count in freq.items():
            if count > max_count:
                most_common_label = label
                max_count = count

        return most_common_label


    def predict(self, X):
        """
        Predicts class labels for the input data using the trained decision tree.
        Parameters:
            X (pd.DataFrame or np.ndarray): Feature data for prediction (shape: n_samples, n_features).
        Returns:
            np.ndarray: Array of predicted class labels (shape: n_samples).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._traverse_tree(sample, self.root) for sample in X])


    def _traverse_tree(self, sample, node):
        """
        Recursively navigates the decision tree to predict a class for a single sample.
        Parameters:
            sample (np.ndarray): Feature values for a single sample (shape: n_features).
            node (Node): Current node in the decision tree.
        Returns:
            any: Predicted class label for the sample.
        """
        if node.is_leaf():
            return node.leaf_value

        feat_idx = node.feature_index
        
        if node.categorical:
            if sample[feat_idx] == node.threshold:
                return self._traverse_tree(sample, node.left)
            else:
                return self._traverse_tree(sample, node.right)
        else:
            if sample[feat_idx] <= node.threshold:
                return self._traverse_tree(sample, node.left)
            else:
                return self._traverse_tree(sample, node.right)


    def predict_and_evaluate(self, X, y):
        predictions = self.predict(X).ravel()
        true_labels = y.values.ravel()
        correct = np.sum(predictions == true_labels)
        errors = np.sum(predictions != true_labels)
        total_samples = len(true_labels)
        loss = errors / total_samples
        return predictions, correct, total_samples, loss
    

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


class TreeVisualizer:

    def __init__(self, tree):
        """
        Initializes the TreeVisualizer for rendering a decision tree.
        Parameters:
            tree (TreePredictor or RandomTreePredictor): Trained decision tree model to visualize.
        """
        self.tree = tree
        self.graph = graphviz.Digraph(format='pdf')
    

    def draw_tree(self, filename='decision_tree'):
        """
        Generates and saves an image of the decision tree as a PNG file.
        Parameters:
            filename (str, optional): Name of the output file (without extension). Default is 'decision_tree'.
        """
        self._add_nodes_edges(self.tree.root)
        self.graph.render(filename, directory='imgs', cleanup=True)
    

    def _add_nodes_edges(self, node, parent_name=None, edge_label=None):
        """
        Adds nodes and edges to the Graphviz object for visualization.
        Parameters:
            node (Node): Current node to add to the graph.
            parent_name (str, optional): Identifier of the parent node for edge creation. 
            edge_label (str, optional): Label for the edge (e.g., threshold condition). 
        """
        if node is None:
            return
        
        node_label = self._get_node_label(node)
        node_name = str(id(node))
        
        self.graph.node(node_name, label=node_label, shape='box' if node.is_leaf() else 'ellipse')
        
        if parent_name is not None:
            self.graph.edge(parent_name, node_name, label=edge_label)
        
        if not node.is_leaf():
            if node.categorical:
                self._add_nodes_edges(node.left, node_name, f"= {node.threshold}")
                self._add_nodes_edges(node.right, node_name, f"!= {node.threshold}")
            else:
                self._add_nodes_edges(node.left, node_name, f"<= {node.threshold}")
                self._add_nodes_edges(node.right, node_name, f"> {node.threshold}")
    

    def _get_node_label(self, node):
        """
        Generates the label for a node for visualization purposes.
        Parameters:
            node (Node): Node for which to create a label.
        Returns:
            str: Label describing the node’s feature, threshold, and gain (for split nodes) or class (for leaf nodes).
        """
        if node.is_leaf():
            return f"Leaf\nClass: {node.leaf_value}"
        return f"{node.feature_name}\nThreshold: {node.threshold}\nGain: {node.info_gain:.4f}"


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


class RandomTreePredictor(TreePredictor):

    def __init__(self, max_features=None, *args, **kwargs):
        """
        Initializes a RandomTreePredictor with feature subsampling, extending TreePredictor.
        Parameters:
            max_features (int or None, optional): Number of features to consider for each split. If None, uses all features. Default is None.
            *args: Positional arguments passed to the parent TreePredictor class.
            **kwargs: Keyword arguments passed to the parent TreePredictor class (e.g., min_samples_split, max_depth, criterion).
        """
        super().__init__(*args, **kwargs)
        self.max_features = max_features


    def _find_best_split(self, X, y):
        """
        Identifies the optimal feature and threshold for splitting, using a random subset of features if specified.
        Parameters:
            X (np.ndarray): NumPy array of feature data (shape: n_samples, n_features).
            y (np.ndarray): NumPy array of target labels (shape: n_samples).
        Returns:
            best_feat (int or None): Index of the best feature to split on, or None if no valid split.
            best_thresh (float or str or None): Optimal threshold (numerical or categorical), or None if no valid split.
            best_gain (float): Information gain achieved, or -inf if no valid split.
            best_left_idxs (np.ndarray or None): Indices of samples in the left branch, or None if no valid split.
            best_right_idxs (np.ndarray or None): Indices of samples in the right branch, or None if no valid split.
        """
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        best_left_idxs, best_right_idxs = None, None
        current_impurity = self._calc_impurity(y)

        # Determine features to consider
        if self.max_features is not None and self.max_features < self.n_features:
            features_to_consider = np.random.choice(self.n_features, size=self.max_features, replace=False)
        else:
            features_to_consider = range(self.n_features)

        # Iterate over the random subset of features to find the optimal split
        for feat_idx in features_to_consider:
            X_column = X[:, feat_idx]
            values = np.unique(X_column)
            for thresh in values:
                left_idxs, right_idxs = self._split_data(X_column, thresh, feat_idx)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                gain = self._calc_info_gain(y, left_idxs, right_idxs, current_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
                    best_left_idxs, best_right_idxs = left_idxs, right_idxs

        return best_feat, best_thresh, best_gain, best_left_idxs, best_right_idxs
    

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


class RandomForest:

    def __init__(self, n_estimators=20, max_depth=30, min_samples_split=10, criterion="gini", max_features="sqrt", random_state=None, n_jobs=-1):
        """
        Initializes a RandomForest classifier with multiple decision trees.
        Parameters:
            n_estimators (int, optional): Number of trees in the forest. Default is 20.
            max_depth (int or None, optional): Maximum depth of each tree, or None for unlimited depth. Default is 30.
            min_samples_split (int, optional): Minimum number of samples required to split a node. Default is 10.
            criterion (str, optional): Splitting criterion ("gini", "entropy", "MSE"). Default is "gini".
            max_features (int, float, str, or None, optional): Number of features to consider per split. Options are:
                - int: Fixed number of features.
                - float: Fraction of total features.
                - "sqrt": Square root of total features.
                - "log2": Log base 2 of total features.
                - None: All features. Default is "sqrt".
            random_state (int or None, optional): Seed for random number generation, ensuring reproducibility. Default is None.
            n_jobs (int, optional): Number of parallel jobs for training (-1 uses all processors, 1 is sequential). Default is 1.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.max_features_ = None  # Set during fit


    def _train_single_tree(self, X, y, indices, tree_idx):
        """
        Trains a single RandomTreePredictor on a bootstrapped sample.
        Parameters:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target labels.
            indices (np.ndarray): Indices for the bootstrapped sample.
            tree_idx (int): Index of the tree for seed generation.
        Returns:
            RandomTreePredictor: Trained tree.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state + tree_idx)

        # Bootstrap sampling
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y.iloc[indices]

        # Create and train a random tree
        tree = RandomTreePredictor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features_
        )
        tree.fit(X_bootstrap, y_bootstrap)
        return tree


    def fit(self, X, y):
        """
        Trains the RandomForest classifier by building multiple decision trees in parallel.
        Parameters:
            X (pd.DataFrame): Feature data for training (shape: n_samples, n_features).
            y (pd.Series): Target labels for training (shape: n_samples).
        """
        print(f'Starting training of the Random Forest model using \n\t- Number of estimators = {self.n_estimators} \
              \n\t- Number of features to consider per split = {self.max_features}({X.shape[1]}) = {max(1, int(np.sqrt(X.shape[1])))} \
              \n\t- Splitting criterion = {self.criterion}\n\t- Maximum depth = {self.max_depth} \
              \n\t- Minimum number of samples required in a node for splitting = {self.min_samples_split} ...')

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Compute the number of features to consider at each split
        if self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            self.max_features_ = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            self.max_features_ = max(1, self.max_features)
        elif isinstance(self.max_features, float):
            self.max_features_ = max(1, int(self.max_features * n_features))
        else:
            self.max_features_ = n_features

        # Generate bootstrap indices for all trees
        bootstrap_indices = [
            np.random.choice(n_samples, size=n_samples, replace=True)
            for _ in range(self.n_estimators)
        ]

        # Train trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_tree)(X, y, indices, idx)
            for idx, indices in enumerate(bootstrap_indices)
        )

        print(f"Random Forest with {self.n_estimators} trees successfully fitted!")


    def predict(self, X):
        """
        Predicts class labels for the input data using majority voting across all trees.
        Parameters:
            X (pd.DataFrame or np.ndarray): Feature data for prediction (shape: n_samples, n_features).
        Returns:
            np.ndarray: Array of predicted class labels (shape: n_samples).
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T

        def majority_vote(preds):
            freq = {}
            for p in preds:
                if p in freq:
                    freq[p] += 1
                else:
                    freq[p] = 1
            
            max_label = None
            max_count = 0
            for label, count in freq.items():
                if count > max_count:
                    max_label = label
                    max_count = count
            return max_label

        predictions = [majority_vote(sample_preds) for sample_preds in tree_preds]
        return np.array(predictions)


    def predict_and_evaluate(self, X, y):
        """
        Predicts class labels and evaluates performance against true labels.
        Parameters:
            X (pd.DataFrame or np.ndarray): Feature data for prediction (shape: n_samples, n_features).
            y (np.ndarray or pd.Series): True class labels (shape: n_samples). 
        Returns:
            tuple: (predictions, correct_predictions, total_predictions, loss)
                - predictions: Array of predicted class labels (shape: n_samples).
                - correct_predictions: Number of correct predictions.
                - total_predictions: Total number of predictions (n_samples).
                - loss: 0-1 loss (fraction of incorrect predictions).
        """
       
        y = np.squeeze(y)
        predictions = self.predict(X)
        total_predictions = len(y)
        correct_predictions = np.sum(predictions == y)
        loss = 1 - (correct_predictions / total_predictions) if total_predictions > 0 else 1.0
        
        return predictions, correct_predictions, total_predictions, loss


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


def print_feature_importance(model, model_type, model_name):
    """
    Prints the feature importance for a TreePredictor or RandomForest model.
    Parameters:
        model: Instance of TreePredictor or RandomForest with feature_importance_ computed.
        model_type (str): Type of model, either 'tree' or 'forest'.
    """
    if model_type == 'tree':
        importance = model.feature_importance_
        feature_names = model.feature_names
    elif model_type == 'forest':
        importance = np.mean([tree.feature_importance_ for tree in model.trees], axis=0)    # Average feature importance across all trees
        feature_names = model.trees[0].feature_names
    else:
        raise ValueError("model_type must be 'tree' or 'forest'")

    # Sorted visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print(f'\nFeature Importance ({model_name}):')
    print(importance_df.to_string(index=False))