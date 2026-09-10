from pathlib import Path

import numpy as np
import torch

from world_model.hybrid_world_model import (
    HybridWorldModel,
)


class HybridWorldModelWrapper:
    """
    Safe inference wrapper around the Hybrid World Model.

    The wrapper handles:
    - device selection
    - tensor conversion
    - prediction
    - saving
    - loading
    """

    def __init__(
        self,
        state_dimension,
        graph_input_dimension=6,
        hidden_dimension=64,
        lstm_layers=1,
        residual_scale=0.75,
        device=None,
    ):

        self.state_dimension = int(
            state_dimension
        )
        self.graph_input_dimension = int(
            graph_input_dimension
        )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = HybridWorldModel(
            state_dimension=self.state_dimension,
            graph_input_dimension=self.graph_input_dimension,
            hidden_dimension=hidden_dimension,
            lstm_layers=lstm_layers,
            residual_scale=residual_scale,
        ).to(self.device)

        self.model.eval()

    def predict(
        self,
        state_sequence,
        graph_sequence=None,
    ):
        """
        Predict the next network state.

        Parameters
        ----------
        state_sequence:
            Shape:
            (sequence_length, state_dimension)

        graph_sequence:
            Optional graph sequence.

        Returns
        -------
        numpy.ndarray
        """

        states = np.asarray(
            state_sequence,
            dtype=np.float32,
        )

        if states.ndim == 2:
            states = states[None, ...]

        if states.ndim != 3:
            raise ValueError(
                "state_sequence must have shape "
                "(sequence_length, state_dimension) "
                "or "
                "(batch, sequence_length, state_dimension)."
            )

        if states.shape[-1] != self.state_dimension:
            raise ValueError(
                f"Expected state dimension "
                f"{self.state_dimension}, "
                f"received {states.shape[-1]}."
            )

        state_tensor = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        if graph_sequence is None:
            graph_sequence = [
                []
                for _ in range(
                    states.shape[0]
                )
            ]

        # The model expects:
        #
        # batch -> sequence -> graph
        #
        # For one prediction:
        #
        # [graph_1, graph_2, ...]
        #
        # becomes:
        #
        # [[graph_1, graph_2, ...]]

        if (
            isinstance(graph_sequence, list)
            and len(graph_sequence) > 0
            and not isinstance(
                graph_sequence[0],
                list,
            )
        ):
            graph_sequence = [
                graph_sequence
            ]

        with torch.no_grad():

            prediction = self.model(
                state_tensor,
                graph_sequence,
            )

        if isinstance(prediction, tuple):
            prediction = prediction[0]

        prediction = prediction.detach()

        if prediction.ndim == 2:
            prediction = prediction[0]

        return prediction.cpu().numpy()

    def predict_tensor(
        self,
        state_tensor,
        graph_sequence=None,
    ):

        if not torch.is_tensor(
            state_tensor
        ):
            state_tensor = torch.tensor(
                state_tensor,
                dtype=torch.float32,
            )

        state_tensor = state_tensor.to(
            self.device
        )

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():

            prediction = self.model(
                state_tensor,
                graph_sequence,
            )

        return prediction

    def save(self, path):
        """
        Save model checkpoint.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "state_dimension":
                self.state_dimension,

            "model_state_dict":
                self.model.state_dict(),

        }

        torch.save(
            checkpoint,
            path,
        )

    def load(
        self,
        path,
        strict=True,
    ):
        """
        Load model checkpoint into this wrapper.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"World Model checkpoint "
                f"not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if isinstance(
            checkpoint,
            dict
        ):

            state_dimension = checkpoint.get(
                "state_dimension"
            )

            if (
                state_dimension is not None
                and int(state_dimension)
                != self.state_dimension
            ):
                raise ValueError(
                    "Checkpoint state dimension "
                    f"{state_dimension} does not match "
                    f"current state dimension "
                    f"{self.state_dimension}."
                )

            state_dict = checkpoint.get(
                "model_state_dict"
            )

            if state_dict is None:

                state_dict = checkpoint.get(
                    "state_dict"
                )

        else:
            state_dict = checkpoint

        if state_dict is None:
            raise ValueError(
                "Invalid World Model checkpoint."
            )

        self.model.load_state_dict(
            state_dict,
            strict=strict,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        return self

    @classmethod
    def from_checkpoint(
        cls,
        path,
        state_dimension=None,
        device=None,
    ):
        """
        Construct wrapper directly from a checkpoint.

        This avoids the previous incorrect pattern:

            HybridWorldModelWrapper.load(...)
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=(
                device
                if device is not None
                else "cpu"
            ),
        )

        checkpoint_dimension = None

        if isinstance(
            checkpoint,
            dict
        ):
            checkpoint_dimension = checkpoint.get(
                "state_dimension"
            )

        if state_dimension is None:

            if checkpoint_dimension is None:
                raise ValueError(
                    "state_dimension must be supplied "
                    "when the checkpoint does not contain it."
                )

            state_dimension = int(
                checkpoint_dimension
            )

        graph_input_dimension = 6
        if isinstance(checkpoint, dict):
            graph_input_dimension = int(
                checkpoint.get("graph_input_dimension", 6)
            )

        wrapper = cls(
            state_dimension=state_dimension,
            graph_input_dimension=graph_input_dimension,
            device=device,
        )

        wrapper.load(
            path
        )

        return wrapper