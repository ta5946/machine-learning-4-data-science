import csv
import numpy as np
import random
import unittest


# Test your implementation with unit tests that focus on the critical or hard parts and edge cases
class MyTests(unittest.TestCase):

    # Tester A
    def setUp(self):
        # Simple linearly separable data
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 0, 1, 1])

    def test_tree_fits_training_data(self):
        # Full tree should perfectly fit the training data
        m = Tree(rand=random.Random(0), min_samples=2).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_pure_node_returns_class(self):
        # Pure node should return the only class present without splitting
        X_pure = np.array([[0, 0], [0, 1]])
        y_pure = np.array([1, 1])
        m = Tree(rand=random.Random(0)).build(X_pure, y_pure)
        np.testing.assert_equal(m.predict(X_pure), y_pure)

    def test_min_samples_stops_splitting(self):
        # Tree should not split if min_samples exceeds the number of instances
        m = Tree(rand=random.Random(0), min_samples=5).build(self.X, self.y)
        preds = m.predict(self.X)
        # All predictions should be the same majority class
        self.assertTrue(np.all(preds == preds[0]))

    def test_min_samples_one_fits(self):
        # Tree should build and predict correctly with min_samples=1
        m = Tree(rand=random.Random(0), min_samples=1).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_single_feature_multiclass(self):
        # Tree should work with a single feature column and multiple distinct classes
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 1, 2, 3])
        m = Tree(rand=random.Random(0)).build(X, y)
        np.testing.assert_equal(m.predict(X), y)

    def test_no_valid_split_returns_prediction(self):
        # Tree should handle the case where no valid split exists gracefully
        X = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        m = Tree(rand=random.Random(0)).build(X, y)
        self.assertEqual(len(m.predict(X)), len(y))

    def test_negative_features(self):
        # Tree should handle feature matrices containing negative values
        X_neg = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        m = Tree(rand=random.Random(0)).build(X_neg, self.y)
        np.testing.assert_equal(m.predict(X_neg), self.y)

    def test_predict_single_sample(self):
        # Prediction of a single instance should return an array of length 1
        m = Tree(rand=random.Random(0)).build(self.X, self.y)
        self.assertEqual(len(m.predict(self.X[0:1])), 1)

    def test_predict_1d_input_raises(self):
        # Predict should raise an error when given a 1D array instead of 2D
        m = Tree(rand=random.Random(0)).build(self.X, self.y)
        with self.assertRaises(IndexError):
            m.predict(np.array([0, 0]))

    # Tester B
    def test_rf_fits_training_data(self):
        # Random forest with enough trees should perfectly fit simple training data
        m = RandomForest(rand=random.Random(0), n=50).build(self.X, self.y)
        np.testing.assert_equal(m.predict(self.X), self.y)

    def test_rf_pure_target(self):
        # Random forest should predict correctly when all training labels are the same
        y_pure = np.array([1, 1, 1, 1])
        m = RandomForest(rand=random.Random(0), n=5).build(self.X, y_pure)
        np.testing.assert_equal(m.predict(self.X), y_pure)

    def test_rf_predict_single_sample(self):
        # Random forest prediction of a single instance should return an array of length 1
        m = RandomForest(rand=random.Random(0), n=10).build(self.X, self.y)
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


# X is a 2D array, rows are instances, columns are features
def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    # Return a random subset of column indices of size sqrt
    c = rand.sample(range(X.shape[1]), int(np.sqrt(X.shape[1])))
    return c


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
        return (best_col, best_thresh, left, right)

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
        if not isinstance(node, tuple):
            return node
        # Otherwise compare the instance column value to the learned threshold
        col, thresh, left, right = node
        if x[col] <= thresh:
            # And recursively call the prediction function on either the left or right node
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)


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
        n_samples = X.shape[0]

        for _ in range(self.n):
            # Get a bootstrap sample of the data (with replacement)
            indices = self.rand.choices(range(n_samples), k=n_samples)
            X_boot, y_boot = X[indices], y[indices]
            # Build a decision tree with it
            trees.append(self.rftree.build(X_boot, y_boot))

        # Construct a random forest from the list of trees
        # Return a RFModel object used for prediction
        return RFModel(trees)


class RFModel:

    def __init__(self, trees):
        self.trees = trees  # List of TreeModel objects

    def predict(self, X):
        # Collect predictions from each tree for each instance in X
        # Here, single tree prediction is a row
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the majority vote for each instance as the final prediction
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )  # Apply count argmax along the first axis

    # Leave for part 2
    def importance(self):
        pass


def misclassification_rate(y_true, y_pred):
    # Count the number of misclassified instances
    errors = y_true != y_pred
    # Error rate = n_errors / n_total
    rate = np.mean(errors)
    # Standard error = sqrt(rate * (1 - rate) / n_total) for Bernoulli distribution
    se = np.sqrt(rate * (1 - rate) / len(y_true))
    return float(rate), float(se)


def hw_tree_full(learn, test):
    # Build a tree with min_samples=2
    X_train, y_train = learn
    X_test, y_test = test

    tree = Tree(rand=random.Random(0), min_samples=2)
    tree_model = tree.build(X_train, y_train)

    # Return misclassification rates and standard errors when using training and testing data as test sets
    train_rate, train_se = misclassification_rate(y_train, tree_model.predict(X_train))
    test_rate, test_se = misclassification_rate(y_test, tree_model.predict(X_test))

    return (train_rate, train_se), (test_rate, test_se)


def hw_randomforests(learn, test):
    # Use random forests with n=100 trees with min_samples=2
    X_train, y_train = learn
    X_test, y_test = test

    rf = RandomForest(rand=random.Random(0), n=100)
    rf_model = rf.build(X_train, y_train)

    # Return misclassification rates and standard errors when using training and testing data as test sets
    train_rate, train_se = misclassification_rate(y_train, rf_model.predict(X_train))
    test_rate, test_se = misclassification_rate(y_test, rf_model.predict(X_test))

    return (train_rate, train_se), (test_rate, test_se)


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


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Load the dataset and print its size
    learn, test, legend = tki()
    print(f"Dataset size: {learn[0].shape[0]} train, {test[0].shape[0]} test")

    # Evaluate full tree and random forest classifiers
    # Print error rates for train and test sets
    (train_rate, train_se), (test_rate, test_se) = hw_tree_full(learn, test)
    print(
        f"Full Tree:     train={train_rate:.4f}±{train_se:.4f}  test={test_rate:.4f}±{test_se:.4f}"
    )
    (train_rate, train_se), (test_rate, test_se) = hw_randomforests(learn, test)
    print(
        f"Random Forest: train={train_rate:.4f}±{train_se:.4f}  test={test_rate:.4f}±{test_se:.4f}"
    )
