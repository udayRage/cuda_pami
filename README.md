# cuda_pami

A framework for developing and benchmarking data mining algorithms, with a special focus on GPU acceleration using CUDA and the RAPIDS ecosystem.

## Project Overview

The main goal of this project is to provide a structured, high-performance environment for implementing and testing pattern mining algorithms. It leverages `cudf` and `cupy` to harness the power of NVIDIA GPUs for significantly faster data processing compared to traditional CPU-based approaches.

The framework provides a base class (`BaseAlgorithm`) that includes built-in support for:
*   Performance timing
*   Memory profiling
*   A standardized interface for running algorithms

## Getting Started

### Prerequisites

*   An NVIDIA GPU with CUDA installed.
*   Python 3.12+
*   The RAPIDS suite, including `cudf` and `cupy`.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/udayRage/cuda_pami.git
    cd cuda_pami
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    <!-- Or use RAPIDs https://rapids.ai -->
    Or follow the [RAPIDS installation guide](https://rapids.ai/) for more options.


## Developer's Guide

### Adding a New Algorithm

To add a new algorithm to the project, use the `setup_algorithm.sh` script. This script automates the creation of the necessary files and directories for a new algorithm experiment.

**Usage:**

```bash
bash scripts/setup_algorithm.sh <algorithm_type> <algorithm_name>
```

**Arguments:**

*   `<algorithm_type>`: The category of the algorithm (e.g., `fuzzy`, `frequent`, `sequential`).
*   `<algorithm_name>`: The name of the new algorithm (e.g., `MyFuzzyMiner`).

**Example:**

To create a new fuzzy algorithm named `MyFuzzyMiner`, run:

```bash
bash scripts/setup_algorithm.sh fuzzy MyFuzzyMiner
```

This will create the following structure:

*   `src/algorithms/fuzzy/MyFuzzyMiner.py`: The Python source file for your algorithm.
*   `notebooks/fuzzy/MyFuzzyMiner.ipynb`: A Jupyter notebook for experimentation.
*   `results/fuzzy/MyFuzzyMiner/`: A directory to store output files and results.

### Implementing the Algorithm

1.  **Open the generated Python file**: `src/algorithms/fuzzy/MyFuzzyMiner.py`.
2.  **Inherit from `BaseAlgorithm`**: Your new class should inherit from `BaseAlgorithm` to get the profiling and timing features.
3.  **Implement the `_mine` method**: This is where the core logic of your algorithm goes. The `BaseAlgorithm`'s `mine()` method will call your `_mine()` method and handle the performance measurement.
4.  **Add Documentation**: Write clear and concise docstrings for your class and its methods. Explain what the algorithm does, its parameters, and what it returns.

**Example Structure (`MyFuzzyMiner.py`):**

```python
from ..base import BaseAlgorithm
# Import any other necessary libraries like cudf, cupy, etc.

class MyFuzzyMiner(BaseAlgorithm):
    """
    A brief description of what MyFuzzyMiner does.
    """

    def __init__(self, dataset, min_support, debug=False):
        """
        Initializes the MyFuzzyMiner algorithm.

        Args:
            dataset (cudf.DataFrame): The input dataset.
            min_support (float): The minimum support threshold.
            debug (bool): Enables detailed profiling if True.
        """
        super().__init__(debug=debug)
        self.dataset = dataset
        self.min_support = min_support
        # ... other initializations

    def _mine(self):
        """
        The core logic for the MyFuzzyMiner algorithm.
        This method should populate the `self.patterns` dictionary.
        """
        print("Running MyFuzzyMiner logic...")
        # --- Your algorithm logic goes here ---
        # Example:
        # frequent_itemsets = self.dataset.do_something()
        # self.patterns = self._format_patterns(frequent_itemsets)
        # --- End of algorithm logic ---

    def save(self, filepath):
        """
        Saves the found patterns to a file.

        Args:
            filepath (str): The path to the output file.
        """
        print(f"Saving patterns to {filepath}...")
        # Implement the logic to save your patterns
        with open(filepath, 'w') as f:
            for pattern, items in self.patterns.items():
                f.write(f"{pattern}: {items}\n")
        print("Patterns saved.")
```

### Experimenting with the Notebook

Use the generated Jupyter Notebook (`notebooks/fuzzy/MyFuzzyMiner.ipynb`) to:
1.  Load your dataset (preferably into a `cudf` DataFrame).
2.  Instantiate your algorithm class.
3.  Run the `mine()` method to execute the algorithm.
4.  Analyze and visualize the results stored in `self.patterns`.
5.  Save the results using the `save()` method.
