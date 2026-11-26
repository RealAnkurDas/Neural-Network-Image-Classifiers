import unittest

import numpy as np

from train_pneumonia_ann import initialize_parameters, nn_cost_function, nn_gradient


class NeuralNetworkGradientTests(unittest.TestCase):
    def test_backpropagation_matches_numerical_gradient(self):
        rng = np.random.default_rng(2)
        features = rng.random((4, 3))
        labels = np.array([1, 2, 1, 2])
        hidden_size = 2
        l2 = 0.3
        parameters = initialize_parameters(features.shape[1], hidden_size, rng)

        analytical = nn_gradient(parameters, features, labels, hidden_size, l2)
        numerical = np.zeros_like(analytical)
        epsilon = 1e-5
        for index in range(len(parameters)):
            perturbation = np.zeros_like(parameters)
            perturbation[index] = epsilon
            cost_plus = nn_cost_function(
                parameters + perturbation, features, labels, hidden_size, l2
            )
            cost_minus = nn_cost_function(
                parameters - perturbation, features, labels, hidden_size, l2
            )
            numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
