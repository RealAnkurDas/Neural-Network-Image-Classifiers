"""Train a handwritten dense ANN to classify NORMAL and PNEUMONIA X-rays."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import optimize

from dataset_loader import CLASS_NAMES, load_group_splits


NUM_LABELS = len(CLASS_NAMES)
DEFAULT_IMAGE_SIZE = 10
DEFAULT_HIDDEN_SIZE = 10
DEFAULT_L2 = 1.02
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_SEED = 42


def _unpack_parameters(
    parameters: np.ndarray,
    input_size: int,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_end = hidden_size * (input_size + 1)
    second_end = first_end + hidden_size * (hidden_size + 1)
    theta1 = parameters[:first_end].reshape(hidden_size, input_size + 1)
    theta2 = parameters[first_end:second_end].reshape(hidden_size, hidden_size + 1)
    theta3 = parameters[second_end:].reshape(NUM_LABELS, hidden_size + 1)
    return theta1, theta2, theta3


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))


def forward_propagation(
    parameters: np.ndarray,
    features: np.ndarray,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    theta1, theta2, theta3 = _unpack_parameters(
        parameters, features.shape[1], hidden_size
    )
    activation1 = np.column_stack((np.ones(features.shape[0]), features))
    activation2 = np.column_stack(
        (np.ones(features.shape[0]), sigmoid(activation1 @ theta1.T))
    )
    activation3 = np.column_stack(
        (np.ones(features.shape[0]), sigmoid(activation2 @ theta2.T))
    )
    probabilities = sigmoid(activation3 @ theta3.T)
    return (
        probabilities,
        activation1,
        activation2,
        activation3,
        (theta1, theta2, theta3),
    )


def nn_cost_function(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    hidden_size: int,
    l2: float,
) -> float:
    probabilities, _, _, _, thetas = forward_propagation(
        parameters, features, hidden_size
    )
    targets = np.eye(NUM_LABELS)[labels - 1]
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    data_cost = -np.sum(
        targets * np.log(probabilities)
        + (1 - targets) * np.log(1 - probabilities)
    ) / len(labels)
    regularization = sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)
    return float(data_cost + l2 * regularization / (2 * len(labels)))


def nn_gradient(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    hidden_size: int,
    l2: float,
) -> np.ndarray:
    probabilities, activation1, activation2, activation3, thetas = (
        forward_propagation(parameters, features, hidden_size)
    )
    theta1, theta2, theta3 = thetas
    targets = np.eye(NUM_LABELS)[labels - 1]
    sample_count = len(labels)

    delta4 = probabilities - targets
    hidden2 = activation3[:, 1:]
    delta3 = (delta4 @ theta3[:, 1:]) * hidden2 * (1 - hidden2)
    hidden1 = activation2[:, 1:]
    delta2 = (delta3 @ theta2[:, 1:]) * hidden1 * (1 - hidden1)

    grad1 = delta2.T @ activation1 / sample_count
    grad2 = delta3.T @ activation2 / sample_count
    grad3 = delta4.T @ activation3 / sample_count
    for gradient, theta in ((grad1, theta1), (grad2, theta2), (grad3, theta3)):
        gradient[:, 1:] += l2 * theta[:, 1:] / sample_count

    return np.concatenate((grad1.ravel(), grad2.ravel(), grad3.ravel()))


def initialize_parameters(
    input_size: int,
    hidden_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    def initialize_layer(fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_out, fan_in + 1))

    theta1 = initialize_layer(input_size, hidden_size)
    theta2 = initialize_layer(hidden_size, hidden_size)
    theta3 = initialize_layer(hidden_size, NUM_LABELS)
    return np.concatenate((theta1.ravel(), theta2.ravel(), theta3.ravel()))


def predict(
    parameters: np.ndarray,
    features: np.ndarray,
    hidden_size: int,
) -> np.ndarray:
    probabilities, _, _, _, _ = forward_propagation(
        parameters, features, hidden_size
    )
    return np.argmax(probabilities, axis=1) + 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch ANN for NORMAL vs PNEUMONIA X-rays."
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.image_size < 1 or args.hidden_size < 1 or args.max_iterations < 1:
        parser.error("image-size, hidden-size, and max-iterations must be positive")
    if args.l2 < 0:
        parser.error("l2 must be non-negative")

    dataset_root = Path(__file__).resolve().parent / "data" / "archive"
    print("Loading images and creating source-group splits...", flush=True)
    splits = load_group_splits(
        dataset_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    train = splits["train"]
    validation = splits["val"]
    test = splits["test"]

    for split_name, batch in splits.items():
        class_counts = {
            class_name: int(np.count_nonzero(batch.labels == label))
            for label, class_name in enumerate(CLASS_NAMES, start=1)
        }
        source_group_count = len(set(batch.group_ids))
        print(
            f"{split_name.title()}: {len(batch.labels)} images, "
            f"{source_group_count} source groups, {class_counts}"
        )

    rng = np.random.default_rng(args.seed)
    initial_parameters = initialize_parameters(
        train.features.shape[1], args.hidden_size, rng
    )
    cost_args = (train.features, train.labels, args.hidden_size, args.l2)
    initial_cost = nn_cost_function(initial_parameters, *cost_args)
    parameters, final_cost, function_calls, gradient_calls, status = optimize.fmin_cg(
        nn_cost_function,
        initial_parameters,
        fprime=nn_gradient,
        args=cost_args,
        maxiter=args.max_iterations,
        full_output=True,
        disp=False,
    )

    status_text = {
        0: "converged",
        1: "maximum iterations reached; convergence not confirmed",
        2: "precision loss; convergence not confirmed",
    }.get(status, f"stopped with optimizer status {status}")
    print(f"Optimizer: {status_text}")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Final cost: {final_cost:.6f}")
    print(
        f"Function evaluations: {function_calls}; "
        f"gradient evaluations: {gradient_calls}"
    )
    print(
        f"Parameters: image_size={args.image_size}, "
        f"hidden_size={args.hidden_size}, l2={args.l2}, seed={args.seed}"
    )

    for split_name, batch in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        accuracy = np.mean(
            predict(parameters, batch.features, args.hidden_size) == batch.labels
        )
        print(f"{split_name} accuracy: {accuracy * 100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
