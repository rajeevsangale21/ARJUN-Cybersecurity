import numpy as np


class TemporalExplainer:
    """
    Explains the importance of historical states
    in a World Model prediction.

    This is useful for answering:

        "Which previous time window influenced
         the prediction the most?"
    """

    def __init__(self, world_model):
        self.world_model = world_model

    def explain_sequence(
        self,
        sequence,
        top_k=None
    ):
        """
        Estimate importance of each historical
        timestep.

        Parameters
        ----------
        sequence:
            Shape:
            (sequence_length, state_dimension)

        top_k:
            Number of important time steps to return.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2:
            raise ValueError(
                "Sequence must have shape "
                "(sequence_length, state_dimension)."
            )

        baseline = self.world_model.predict(
            sequence
        )

        importance = []

        for timestep in range(
            len(sequence)
        ):

            modified_sequence = (
                sequence.copy()
            )

            # Replace the selected historical
            # state with the mean state of the
            # sequence.
            replacement = np.mean(
                sequence,
                axis=0
            )

            modified_sequence[
                timestep
            ] = replacement

            modified_prediction = (
                self.world_model.predict(
                    modified_sequence
                )
            )

            change = np.mean(
                np.abs(
                    modified_prediction
                    -
                    baseline
                )
            )

            importance.append(
                float(change)
            )

        importance = np.asarray(
            importance,
            dtype=np.float32
        )

        total = importance.sum()

        if total > 0:

            normalized = (
                importance / total
            )

        else:

            normalized = (
                np.zeros_like(
                    importance
                )
            )

        indices = np.argsort(
            normalized
        )[::-1]

        if top_k is not None:
            indices = indices[:top_k]

        results = []

        for index in indices:

            results.append(
                {
                    "timestep":
                        int(index),

                    "importance":
                        float(
                            normalized[index]
                        )
                }
            )

        return results