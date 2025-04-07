import numpy as np


class FirstOrderMarkovChain:
    def __init__(self, n_states=4):
        """
        Initialize a first-order Markov chain model

        Parameters:
        n_states: Number of states, default is 4 (corresponding to states 1,2,3,4)
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.initial_distribution = None
        self.transition_counts = None
        self.initial_counts = None

    def fit(self, sequences, smoothing=1.0):
        """
        Fit the Markov chain model based on given sequences

        Parameters:
        sequences: List of sequences, each sequence is a list containing 1,2,3,4
        smoothing: Laplace smoothing parameter to avoid zero probability issues
        """
        # Initialize transition count matrix and initial state counts
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        self.initial_counts = np.zeros(self.n_states)

        # Count transition frequencies and initial state frequencies
        for sequence in sequences:
            if len(sequence) == 0:
                continue

            # Record initial state
            self.initial_counts[sequence[0] - 1] += 1

            # Record transitions
            for i in range(len(sequence) - 1):
                current_state = sequence[i] - 1  # Convert to 0-indexed
                next_state = sequence[i + 1] - 1  # Convert to 0-indexed
                self.transition_counts[current_state, next_state] += 1

        # Apply Laplace smoothing and calculate probabilities
        self.transition_counts += smoothing
        self.initial_counts += smoothing

        # Calculate transition probability matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = np.sum(self.transition_counts[i])
            if row_sum > 0:
                self.transition_matrix[i] = self.transition_counts[i] / row_sum
            else:
                # If a state never appears, assume uniform distribution
                self.transition_matrix[i] = np.ones(self.n_states) / self.n_states

        # Calculate initial state distribution
        self.initial_distribution = self.initial_counts / np.sum(self.initial_counts)

        return self

    def generate_sequence(self, length):
        """
        Generate a new sequence

        Parameters:
        length: Length of the sequence to generate

        Returns:
        Generated sequence (elements are 1,2,3,4)
        """
        if self.transition_matrix is None or self.initial_distribution is None:
            raise ValueError("Model not yet fitted, please call the fit method first")

        sequence = []

        # Generate initial state
        current_state = np.random.choice(self.n_states, p=self.initial_distribution)
        sequence.append(current_state + 1)  # Convert to 1-indexed

        # Generate subsequent states
        for _ in range(length - 1):
            current_state = np.random.choice(
                self.n_states, p=self.transition_matrix[current_state]
            )
            sequence.append(current_state + 1)  # Convert to 1-indexed

        return sequence

    def log_likelihood(self, sequences):
        """
        Calculate the log-likelihood of sequences

        Parameters:
        sequences: List of sequences

        Returns:
        Log-likelihood value
        """
        if self.transition_matrix is None or self.initial_distribution is None:
            raise ValueError("Model not yet fitted, please call the fit method first")

        log_likelihood = 0.0

        for sequence in sequences:
            if len(sequence) == 0:
                continue

            # Log probability of initial state
            log_likelihood += np.log(self.initial_distribution[sequence[0] - 1])

            # Log probability of transitions
            for i in range(len(sequence) - 1):
                current_state = sequence[i] - 1
                next_state = sequence[i + 1] - 1
                log_likelihood += np.log(
                    self.transition_matrix[current_state, next_state]
                )

        return log_likelihood

    def print_model(self):
        """Print model parameters"""
        print("Initial state distribution:")
        for i in range(self.n_states):
            print(f"State {i+1}: {self.initial_distribution[i]:.4f}")

        print("\nTransition probability matrix:")
        print("    | " + " ".join([f"  {i+1}  " for i in range(self.n_states)]))
        print("----+" + "------" * self.n_states)
        for i in range(self.n_states):
            print(
                f" {i+1}  | "
                + " ".join(
                    [f"{self.transition_matrix[i,j]:.4f}" for j in range(self.n_states)]
                )
            )

        print("Initial counts:")
        print(self.initial_counts.astype(int))
        print("Transition counts:")
        print(self.transition_counts.astype(int))
