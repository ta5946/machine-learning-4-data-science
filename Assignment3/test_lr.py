import unittest

import numpy as np

from solution1 import MultinomialLogReg, OrdinalLogReg


# UNIT TESTS
class HW2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_multinomial(self):
        l = MultinomialLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        l = OrdinalLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)


class MyTests(unittest.TestCase):

    def setUp(self):
        # Three clearly separated classes along the first feature axis
        X0 = np.column_stack([np.full(20, -3.0), np.zeros((20, 2))])
        X1 = np.column_stack([np.full(20, 0.0), np.zeros((20, 2))])
        X2 = np.column_stack([np.full(20, 3.0), np.zeros((20, 2))])
        self.X = np.vstack([X0, X1, X2])
        self.y = np.array([0] * 20 + [1] * 20 + [2] * 20)

    def test_output_shape_matches_number_of_classes(self):
        # predict() should return one probability column per class
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(self.X, self.y).predict(self.X)
            self.assertEqual(prob.shape, (len(self.X), 3))

    def test_probabilities_sum_to_one(self):
        # Each row must sum to 1
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(self.X, self.y).predict(self.X)
            np.testing.assert_almost_equal(prob.sum(axis=1), np.ones(len(self.X)))

    def test_probabilities_are_in_valid_range(self):
        # No probability should fall outside [0, 1]
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(self.X, self.y).predict(self.X)
            self.assertTrue((prob >= 0).all())
            self.assertTrue((prob <= 1).all())

    def test_predict_on_unseen_data(self):
        # predict() on new samples should return one row per sample
        X_new = np.array([[0.1, -0.2, 0.3], [-1.0, 0.5, 0.0]])
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(self.X, self.y).predict(X_new)
            self.assertEqual(prob.shape, (2, 3))

    def test_predict_single_sample(self):
        # A single-row input should return shape (1, n_classes), not a 1D array
        X_single = np.array([[0.5, -0.5, 1.0]])
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(self.X, self.y).predict(X_single)
            self.assertEqual(prob.shape, (1, 3))
            np.testing.assert_almost_equal(prob.sum(axis=1), [1.0])

    def test_binary_classification(self):
        # Both models should work with only 2 classes (ordinal has just one threshold)
        X_bin = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_bin = np.array([0, 0, 1, 1])
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(X_bin, y_bin).predict(X_bin)
            self.assertEqual(prob.shape, (4, 2))
            np.testing.assert_almost_equal(prob.sum(axis=1), np.ones(4))

    def test_separable_data_mostly_correct(self):
        # With clearly separated classes, argmax of predicted probs should match true labels
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model(lr=0.05, n_steps=3000).build(self.X, self.y).predict(self.X)
            accuracy = np.mean(prob.argmax(axis=1) == self.y)
            self.assertGreater(accuracy, 0.75)

    def test_classes_stored_correctly(self):
        # model.classes should match the sorted unique labels from training
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            model = Model().build(self.X, self.y)
            np.testing.assert_array_equal(model.classes, np.array([0, 1, 2]))

    def test_parameters_updated_after_training(self):
        # Parameters should move away from their zero initialization after gradient descent
        multinomial = MultinomialLogReg(n_steps=50).build(self.X, self.y)
        self.assertFalse(np.all(multinomial.W == 0))
        self.assertFalse(np.all(multinomial.b == 0))

        ordinal = OrdinalLogReg(n_steps=50).build(self.X, self.y)
        self.assertFalse(np.all(ordinal.beta == 0))
        self.assertFalse(np.all(ordinal.thresholds == 0))

    def test_numerical_stability_large_inputs(self):
        # Very large feature values should not produce NaNs
        X_huge = self.X * 1000.0
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model().build(X_huge, self.y).predict(X_huge)
            self.assertFalse(np.isnan(prob).any())
            self.assertTrue((prob >= 0).all() and (prob <= 1).all())

    def test_all_zero_features(self):
        # Zero features give no signal; model should still output valid probabilities
        X_zero = np.zeros_like(self.X)
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model(n_steps=10).build(X_zero, self.y).predict(X_zero)
            self.assertFalse(np.isnan(prob).any())
            np.testing.assert_almost_equal(prob.sum(axis=1), np.ones(len(X_zero)))

    def test_overlapping_data_valid_output(self):
        # Identical inputs with mixed labels give no gradient signal, but outputs must stay valid
        X_overlap = np.zeros((30, 3))
        y_overlap = np.array([0] * 10 + [1] * 10 + [2] * 10)
        for Model in [MultinomialLogReg, OrdinalLogReg]:
            prob = Model(n_steps=100).build(X_overlap, y_overlap).predict(X_overlap)
            self.assertFalse(np.isnan(prob).any())
            np.testing.assert_almost_equal(prob.sum(axis=1), np.ones(len(X_overlap)))


if __name__ == "__main__":
    unittest.main()
