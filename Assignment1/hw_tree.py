import csv
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt
from tqdm import tqdm


# UNIT TESTS
# Test your implementation with unit tests that focus on the critical or hard parts and edge cases
class MyTests(unittest.TestCase):

    # Tester A
    def setUp(self):
        # Simple linearly separable data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 0, 1, 1])

    def test_tree_fits_training_data(self):
        # Full tree should perfectly fit the training data
        m = Tree(rand=random.Random(42), min_samples=2).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_pure_node_returns_class(self):
        # Pure node should return the only class present without splitting
        X_pure = np.array([[0, 0], [0, 1]])
        y_pure = np.array([1, 1])
        m = Tree(rand=random.Random(42)).build(X_pure, y_pure)
        np.testing.assert_equal(m.predict(X_pure), y_pure)

    def test_min_samples_stops_splitting(self):
        # Tree should not split if min_samples exceeds the number of instances
        m = Tree(rand=random.Random(42), min_samples=5).build(self.X, self.y)
        preds = m.predict(self.X)
        # All predictions should be the same majority class
        self.assertTrue(np.all(preds == preds[0]))

    def test_min_samples_one_fits(self):
        # Tree should build and predict correctly with min_samples=1
        m = Tree(rand=random.Random(42), min_samples=1).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_single_feature_multiclass(self):
        # Tree should work with a single feature column and multiple distinct classes
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 1, 2, 3])
        m = Tree(rand=random.Random(42)).build(X, y)
        np.testing.assert_equal(m.predict(X), y)

    def test_no_valid_split_returns_prediction(self):
        # Tree should handle the case where no valid split exists gracefully
        X = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        m = Tree(rand=random.Random(42)).build(X, y)
        self.assertEqual(len(m.predict(X)), len(y))

    def test_negative_features(self):
        # Tree should handle feature matrices containing negative values
        X_neg = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        m = Tree(rand=random.Random(42)).build(X_neg, self.y)
        np.testing.assert_equal(m.predict(X_neg), self.y)

    def test_predict_single_sample(self):
        # Prediction of a single instance should return an array of length 1
        m = Tree(rand=random.Random(42)).build(self.X, self.y)
        self.assertEqual(len(m.predict(self.X[0:1])), 1)

    def test_predict_1d_input_raises(self):
        # Predict should raise an error when given a 1D array instead of 2D
        m = Tree(rand=random.Random(42)).build(self.X, self.y)
        with self.assertRaises(IndexError):
            m.predict(np.array([0, 0]))

    # Tester B
    def test_rf_fits_training_data(self):
        # Random forest with enough trees should perfectly fit simple training data
        m = RandomForest(rand=random.Random(42), n=50).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_rf_pure_target(self):
        # Random forest should predict correctly when all training labels are the same
        y_pure = np.array([1, 1, 1, 1])
        m = RandomForest(rand=random.Random(42), n=5).build(self.X, y_pure)
        np.testing.assert_equal(m.predict(self.X), y_pure)

    def test_rf_predict_single_sample(self):
        # Random forest prediction of a single instance should return an array of length 1
        m = RandomForest(rand=random.Random(42), n=10).build(self.X, self.y)
        self.assertEqual(len(m.predict(self.X[0:1])), 1)

    def test_misclassification_all_correct(self):
        # Perfect predictions should give rate=0 and se=0
        rate, se = misclassification_rate(self.y, self.y)
        self.assertEqual(rate, 0.0)
        self.assertEqual(se, 0.0)

    def test_misclassification_all_wrong(self):
        # Completely wrong predictions should give rate=1 and se=0
        rate, se = misclassification_rate(self.y, 1 - self.y)
        self.assertEqual(rate, 1.0)
        self.assertEqual(se, 0.0)

    def test_misclassification_partial(self):
        # Partial predictions should return the correct rate and a positive standard error
        y_pred = np.array([0, 1, 0, 1])
        rate, se = misclassification_rate(self.y, y_pred)
        self.assertEqual(rate, 0.5)
        self.assertGreater(se, 0.0)

    def test_importance_length(self):
        # Importance should return one value per feature
        m = RandomForest(rand=random.Random(42), n=20).build(
            np.tile(self.X, (2, 1)), np.tile(self.y, 2)
        )
        self.assertEqual(len(m.importance()), self.X.shape[1])

    def test_importance_informative_feature(self):
        # Feature 0 is informative (determines class)
        # It should have higher importance than feature 1 which is just noise
        m = RandomForest(rand=random.Random(42), n=20).build(
            np.tile(self.X, (2, 1)), np.tile(self.y, 2)
        )
        imp = m.importance()
        self.assertGreater(imp[0], imp[1])


# COLUMN SELECTORS
# X is a 2D array, rows are instances, columns are features
def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    # Return a random subset of column indices of size sqrt
    c = rand.sample(range(X.shape[1]), int(np.sqrt(X.shape[1])))
    return c


# TREE NODE
class TreeNode:

    def __init__(self, col, thresh, left, right):
        self.col = col  # Feature index to split on
        self.thresh = thresh  # Threshold for going left vs right
        self.left = left  # Lesser node
        self.right = right  # Greater node


# DECISION TREE
class Tree:

    def __init__(self, rand=None, get_candidate_columns=all_columns, min_samples=2):
        self.rand = rand  # Random generator, for reproducibility
        self.get_candidate_columns = get_candidate_columns  # Function that returns a list of column indices considered for a split
        self.min_samples = min_samples  # Minimum number of samples for which a node is still split further = termination condition

    def build(self, X, y):
        # Construct a decision tree based on the training data X and labels y
        # Return a TreeModel object used for prediction
        return TreeModel(self.build_node(X, y))

    def build_node(self, X, y):
        # Stop at minimum number of samples or pure node
        if len(y) < self.min_samples or len(np.unique(y)) == 1:
            # Return the majority class
            return np.bincount(y).argmax()

        # Find the best split
        best_col, best_thresh = self.best_split(X, y)

        # No valid split was found (for example only 1 unique value)
        if best_col is None:
            # Return the majority class
            return np.bincount(y).argmax()

        # Take indices of rows where column value is less than or equal to threshold for the left node
        left_mask = X[:, best_col] <= best_thresh
        # And greater than for the right node
        right_mask = ~left_mask

        # Avoid empty splits
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.bincount(y).argmax()

        # Recursively build the left and right nodes
        left = self.build_node(X[left_mask], y[left_mask])
        right = self.build_node(X[right_mask], y[right_mask])

        # Return a nested structure used to construct a TreeModel
        return TreeNode(best_col, best_thresh, left, right)

    def best_split(self, X, y):
        best_gini = float("inf")
        best_col = None
        best_thresh = None

        # Either all or a random subset of column indices
        cols = self.get_candidate_columns(X, self.rand)

        # Iterate over candidate columns
        for col in cols:
            # Get unique, sorted column values
            values = np.unique(X[:, col])
            # Get all midpoints by shifting values by one and averaging
            thresholds = (values[:-1] + values[1:]) / 2

            for thresh in thresholds:
                # Take rows where column value is less than or equal to threshold
                left_mask = X[:, col] <= thresh
                right_mask = ~left_mask

                # Avoid one sided splits
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate weighted Gini impurity of the current split
                gini = self.weighted_gini(y[left_mask], y[right_mask])

                # Update the best split (column + threshold) and impurity
                if gini < best_gini:
                    best_gini = gini
                    best_col = col
                    best_thresh = thresh

        # Return the best split
        return best_col, best_thresh

    def gini(self, y):
        # Compute impurity for a list of labels
        # Gini = 1 - sum(p_i^2)
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    def weighted_gini(self, left_y, right_y):
        # Compute impurity for a split so left and right node weighted by size
        # Weighted Gini = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)
        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * self.gini(left_y) + (len(right_y) / n) * self.gini(
            right_y
        )


class TreeModel:

    def __init__(self, tree):
        self.tree = tree  # Nested tuple of (column, threshold, left_node, right_node) or a leaf class value

    def predict(self, X):
        # Predict each instance of X separately
        # Return a 1D array of predictions
        return np.array([self.predict_one(x, self.tree) for x in X])

    def predict_one(self, x, node):
        # If node is a leaf, return its class
        if not isinstance(node, TreeNode):
            return node
        # Otherwise compare the instance column value to the learned threshold
        if x[node.col] <= node.thresh:
            # And recursively call the prediction function on either the left or right node
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)


# RANDOM FOREST
class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n  # Number of bootstrap samples (trees)
        self.rand = rand  # Random generator
        self.rftree = Tree(
            rand=rand, get_candidate_columns=random_sqrt_columns, min_samples=2
        )  # Instance of Tree classifier

    def build(self, X, y):
        # Construct a random forest based on the training data X and labels y
        # Return a RFModel object used for prediction
        trees = []
        oob_indices = []
        n_samples = X.shape[0]

        for _ in range(self.n):
            # Get a bootstrap sample of the data (with replacement)
            indices = self.rand.choices(range(n_samples), k=n_samples)
            # Out of bag instances are not included in the bootstrap sample
            oob = list(set(range(n_samples)) - set(indices))
            X_boot, y_boot = X[indices], y[indices]
            # Build a decision tree with it
            trees.append(self.rftree.build(X_boot, y_boot))
            # Save OOB indices
            oob_indices.append(oob)

        # Construct a random forest from the list of trees
        # Return a RFModel object used for prediction
        return RFModel(trees, oob_indices, X, y, self.rand)


class RFModel:

    def __init__(self, trees, oob_indices, X, y, rand):
        self.trees = trees  # List of TreeModel objects
        self.oob_indices = oob_indices  # List of OOB indices for each tree
        self.X = X  # Training data
        self.y = y  # Training labels
        self.rand = rand  # Random generator for permutations

    def predict(self, X):
        # Collect predictions from each tree for each instance in X
        # Here, single tree prediction is a row
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the majority vote for each instance as the final prediction
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )  # Apply count argmax along the first axis

    # Implement permutation based variable importance
    def importance(self):
        n_vars = self.X.shape[1]
        importances = np.zeros(n_vars)

        # For each tree and its OOB instances
        for tree, oob in zip(self.trees, self.oob_indices):
            # Skip trees with no OOB
            if len(oob) == 0:
                continue

            # Get OOB data and labels
            X_oob = self.X[oob]
            y_oob = self.y[oob]

            # Use this tree to predict OOB instances
            # Calculate baseline accuracy
            baseline = np.mean(tree.predict(X_oob) == y_oob)

            # For each feature
            for var in range(n_vars):
                # Permute its values on OOB data
                X_permuted = X_oob.copy()
                X_permuted[:, var] = self.X[self.rand.sample(oob, len(oob)), var]

                # Calculate accuracy after permutation
                permuted_acc = np.mean(tree.predict(X_permuted) == y_oob)
                # Save the accuracy drop as feature importance
                importances[var] += baseline - permuted_acc

        # Return average importances over all trees in a forest
        importances /= len(self.trees)
        return importances


# ERROR METRICS
def misclassification_rate(y_true, y_pred):
    # Count the number of misclassified instances
    errors = y_true != y_pred
    # Error rate = n_errors / n_total
    rate = np.mean(errors)
    # Standard error = sqrt(rate * (1 - rate) / n_total) for Bernoulli distribution
    se = np.sqrt(rate * (1 - rate) / len(y_true))
    return float(rate), float(se)


# HW PIPELINES
def hw_tree_full(learn, test):
    # Build a tree with min_samples=2
    X_train, y_train = learn
    X_test, y_test = test

    tree = Tree(rand=random.Random(42), min_samples=2)
    tree_model = tree.build(X_train, y_train)

    # Return misclassification rates and standard errors when using training and testing data as test sets
    train_rate, train_se = misclassification_rate(y_train, tree_model.predict(X_train))
    test_rate, test_se = misclassification_rate(y_test, tree_model.predict(X_test))

    return (train_rate, train_se), (test_rate, test_se)


def hw_randomforests(learn, test):
    # Use random forests with n=100 trees with min_samples=2
    X_train, y_train = learn
    X_test, y_test = test

    rf = RandomForest(rand=random.Random(42), n=100)
    rf_model = rf.build(X_train, y_train)

    # Return misclassification rates and standard errors when using training and testing data as test sets
    train_rate, train_se = misclassification_rate(y_train, rf_model.predict(X_train))
    test_rate, test_se = misclassification_rate(y_test, rf_model.predict(X_test))

    return (train_rate, train_se), (test_rate, test_se)


# PLOTS
# Plot misclassification rates versus the number of trees n
def plot_rf_misclassification(learn, test, max_n_trees=101):
    X_train, y_train = learn
    X_test, y_test = test

    n_trees_range = list(range(1, max_n_trees + 1, 5))
    train_rates, test_rates, train_ses, test_ses = [], [], [], []

    # Build random forests with increasing number of trees
    for n_trees in tqdm(n_trees_range, desc="Evaluating forests"):
        rf = RandomForest(rand=random.Random(42), n=n_trees)
        m = rf.build(X_train, y_train)
        # Save misclassification rates and standard errors for train and test sets
        tr, tr_se = misclassification_rate(y_train, m.predict(X_train))
        te, te_se = misclassification_rate(y_test, m.predict(X_test))
        train_rates.append(tr)
        train_ses.append(tr_se)
        test_rates.append(te)
        test_ses.append(te_se)

    train_rates, test_rates = np.array(train_rates), np.array(test_rates)
    train_ses, test_ses = np.array(train_ses), np.array(test_ses)

    # Line plot with shaded uncertainty bands
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_range, train_rates, label="Train")
    plt.fill_between(
        n_trees_range, train_rates - train_ses, train_rates + train_ses, alpha=0.2
    )
    plt.plot(n_trees_range, test_rates, label="Test")
    plt.fill_between(
        n_trees_range, test_rates - test_ses, test_rates + test_ses, alpha=0.2
    )
    plt.xlabel("Number of trees")
    plt.ylabel("Misclassification rate")
    plt.title("Random forest misclassification rate vs number of trees")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rf_misclassification.png")
    plt.show()


# Plot variable importance for the given dataset for an RF with n=100 trees
def plot_variable_importance(learn, legend):
    X_train, y_train = learn
    n_vars = X_train.shape[1]

    # Build a random forest and evaluate feature importance
    rf = RandomForest(rand=random.Random(42), n=100)
    rf_model = rf.build(X_train, y_train)
    importances = rf_model.importance()

    rand = random.Random(42)
    root_counts = np.zeros(n_vars)
    # Build a tree on all columns
    tree = Tree(rand=random.Random(42), min_samples=len(y_train) - 1)
    for _ in tqdm(range(100), desc="Building trees on shuffled data"):
        # Shuffle labels to remove signal
        y_shuffled = y_train.copy()
        rand.shuffle(y_shuffled)
        # We only care about the first split
        node = tree.build_node(X_train, y_shuffled)
        # If exists, save the root split feature
        if isinstance(node, TreeNode):
            root_counts[node.col] += 1

    # Normalize importances and root split counts to frequency
    importances = np.maximum(importances, 0)
    importances /= np.sum(importances)
    root_freq = root_counts / root_counts.sum()

    # Bar plot
    x = np.arange(n_vars)
    plt.figure(figsize=(12, 6))
    plt.bar(x, importances, alpha=0.7, label="RF variable importance")
    plt.bar(x, root_freq, alpha=0.7, label="Tree variable root frequency")
    plt.xlabel("Variable")
    plt.ylabel("Importance")
    plt.title("Variable importance vs its root frequency on randomly shuffled data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("variable_importance.png")
    plt.show()


# DATA UTILS
def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


# MAIN
if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Load the dataset and print its size
    learn, test, legend = tki()
    print(f"Dataset size: {learn[0].shape[0]} train, {test[0].shape[0]} test")

    # Print number of features
    print(f"Total features: {learn[0].shape[1]}")

    # Evaluate full tree and random forest classifiers
    # Print error rates for train and test sets
    (train_rate, train_se), (test_rate, test_se) = hw_tree_full(learn, test)
    print(
        f"Full tree:     train={train_rate:.4f}±{train_se:.4f}  test={test_rate:.4f}±{test_se:.4f}"
    )
    (train_rate, train_se), (test_rate, test_se) = hw_randomforests(learn, test)
    print(
        f"Random forest: train={train_rate:.4f}±{train_se:.4f}  test={test_rate:.4f}±{test_se:.4f}"
    )

    # Plot error rates for increasing number of trees
    plot_rf_misclassification(learn, test)

    # Plot feature importances of a random forest
    plot_variable_importance(learn, legend)
