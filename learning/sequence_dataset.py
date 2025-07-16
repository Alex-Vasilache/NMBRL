import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional


class SequenceDataset(Dataset):
    """
    Custom PyTorch dataset for training RNN world models on sequence data.

    This dataset creates sequences of length `imag_horizon` from the experience buffer.
    Each sequence contains:
    - Input: (state, action) pairs for each timestep
    - Target: (next_state, reward) pairs for each timestep

    The dataset ensures that all sequences are valid (no episode boundaries within a sequence).
    """

    def __init__(
        self,
        experience_data: List[Tuple],
        state_size: int,
        action_size: int,
        imag_horizon: int,
        min_sequence_length: Optional[int] = None,
    ):
        """
        Initialize the sequence dataset.

        Args:
            experience_data: List of (state, action, next_state, reward, done) tuples
            state_size: Size of the state vector
            action_size: Size of the action vector
            imag_horizon: Length of sequences to create
            min_sequence_length: Minimum length of valid sequences (defaults to imag_horizon)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.imag_horizon = imag_horizon
        self.min_sequence_length = min_sequence_length or imag_horizon

        # Extract valid start indices for sequences
        self.valid_start_indices = self._find_valid_sequences(experience_data)

        # Store the experience data
        self.experience_data = experience_data

        print(
            f"[SEQUENCE_DATASET] Created dataset with {len(self.valid_start_indices)} valid sequences"
        )
        print(f"[SEQUENCE_DATASET] Sequence length: {imag_horizon}")
        print(f"[SEQUENCE_DATASET] Total experiences: {len(experience_data)}")

    def _find_valid_sequences(self, experience_data: List[Tuple]) -> List[int]:
        """
        Find all valid start indices for sequences of length imag_horizon.

        A valid sequence is one where:
        1. There are enough consecutive experiences to form a sequence
        2. No episode boundaries (done=True) within the sequence

        Args:
            experience_data: List of (state, action, next_state, reward, done) tuples

        Returns:
            List of valid start indices
        """
        valid_indices = []

        for i in range(len(experience_data) - self.imag_horizon + 1):
            # Check if we have enough consecutive experiences
            if i + self.imag_horizon > len(experience_data):
                break

            # Check if there are any episode boundaries within this sequence
            sequence_valid = True
            for j in range(i, i + self.imag_horizon):
                if j >= len(experience_data):
                    sequence_valid = False
                    break

                # Check if this is the end of an episode (done=True)
                # Note: We don't include the last experience in the sequence if it's done
                if j < i + self.imag_horizon - 1:  # Not the last timestep in sequence
                    _, _, _, _, done = experience_data[j]
                    if done:
                        sequence_valid = False
                        break

            if sequence_valid:
                valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.valid_start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sequence for autoregressive training.

        Returns:
            initial_state: (state_size,)
            action_sequence: (imag_horizon, action_size)
            target_sequence: (imag_horizon, state_size + 1)
        """
        start_idx = self.valid_start_indices[idx]
        sequence_data = self.experience_data[start_idx : start_idx + self.imag_horizon]

        # Initial state is the state at t=0
        initial_state = sequence_data[0][0]
        # Action sequence: all actions in the sequence
        action_sequence = [step[1] for step in sequence_data]
        # Target sequence: (next_state, reward) for each step
        target_sequence = [
            np.concatenate([step[2], [step[3]]]) for step in sequence_data
        ]

        initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32)
        action_tensor = torch.tensor(action_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32)

        return initial_state_tensor, action_tensor, target_tensor

    def get_initial_states(self) -> torch.Tensor:
        """
        Get all initial states from valid sequences for use in model initialization.

        Returns:
            Tensor of shape (num_sequences, state_size) containing initial states
        """
        initial_states = []

        for start_idx in self.valid_start_indices:
            state, _, _, _, _ = self.experience_data[start_idx]
            initial_states.append(state)

        return torch.tensor(initial_states, dtype=torch.float32)

    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        if len(self.valid_start_indices) == 0:
            return {
                "num_sequences": 0,
                "num_experiences": len(self.experience_data),
                "sequence_length": self.imag_horizon,
                "coverage_ratio": 0.0,
            }

        # Calculate coverage ratio (how much of the data is used in valid sequences)
        total_experiences = len(self.experience_data)
        used_experiences = len(self.valid_start_indices) * self.imag_horizon
        coverage_ratio = (
            used_experiences / total_experiences if total_experiences > 0 else 0.0
        )

        return {
            "num_sequences": len(self.valid_start_indices),
            "num_experiences": total_experiences,
            "sequence_length": self.imag_horizon,
            "coverage_ratio": coverage_ratio,
            "valid_start_indices": self.valid_start_indices,
        }


def create_sequence_dataset_from_buffer(
    experience_data: List[Tuple],
    state_size: int,
    action_size: int,
    imag_horizon: int,
    train_val_split: float = 0.8,
    random_seed: int = 42,
) -> Tuple[SequenceDataset, SequenceDataset]:
    """
    Create training and validation sequence datasets from experience data.

    Args:
        experience_data: List of (state, action, next_state, reward, done) tuples
        state_size: Size of the state vector
        action_size: Size of the action vector
        imag_horizon: Length of sequences to create
        train_val_split: Fraction of data to use for training
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create the full dataset
    full_dataset = SequenceDataset(
        experience_data=experience_data,
        state_size=state_size,
        action_size=action_size,
        imag_horizon=imag_horizon,
    )

    # Get valid start indices
    valid_indices = full_dataset.valid_start_indices

    if len(valid_indices) == 0:
        raise ValueError("No valid sequences found in the data")

    # Split indices into train and validation
    np.random.seed(random_seed)
    np.random.shuffle(valid_indices)

    split_idx = int(len(valid_indices) * train_val_split)
    train_indices = valid_indices[:split_idx]
    val_indices = valid_indices[split_idx:]

    # Create train dataset
    train_dataset = SequenceDataset(
        experience_data=experience_data,
        state_size=state_size,
        action_size=action_size,
        imag_horizon=imag_horizon,
    )
    train_dataset.valid_start_indices = train_indices

    # Create validation dataset
    val_dataset = SequenceDataset(
        experience_data=experience_data,
        state_size=state_size,
        action_size=action_size,
        imag_horizon=imag_horizon,
    )
    val_dataset.valid_start_indices = val_indices

    print(f"[SEQUENCE_DATASET] Train sequences: {len(train_indices)}")
    print(f"[SEQUENCE_DATASET] Validation sequences: {len(val_indices)}")

    return train_dataset, val_dataset
