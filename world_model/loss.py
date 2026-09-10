import torch.nn as nn


def world_model_loss(predicted_state, target_state):
    """
    Calculate the World Model transition loss.

    Measures the difference between:

        predicted S_t+1

    and

        actual S_t+1
    """

    criterion = nn.MSELoss()

    return criterion(
        predicted_state,
        target_state
    )