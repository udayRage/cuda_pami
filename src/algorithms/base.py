from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Any
import time
import tracemalloc
import cProfile
import platform

# Import common data science libraries
import pandas as pd
import numpy as np
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # We don't raise an error here anymore, so code can run in CPU-only mode if designed to.
    # Algorithms that require GPU should check `GPU_AVAILABLE`.

# Use resource for lightweight memory profiling on Unix-like systems
if platform.system() != "Windows":
    import resource

# Explicitly export the public API of this module
__all__ = [
    "BaseAlgorithm",
    "GPU_AVAILABLE",
    "pd",
    "np",
    "cudf",
    "cp",
    "Path",
    "Union",
    "Any",
]

class BaseAlgorithm(ABC):
    """
    An abstract base class for all algorithms, providing a consistent framework for
    timing, memory profiling, and pattern management.
    """

    def __init__(self, iFile: Union[str, Path, pd.DataFrame], debug=False, **kwargs):
        """
        Initializes the BaseAlgorithm.
        Args:
            iFile: The input file path or a pandas/cudf DataFrame.
            debug (bool): If True, enables detailed memory and CPU profiling.
            **kwargs: Additional keyword arguments for subclasses.
        """
        self._iFile = iFile
        self.patterns = {}
        self._start_time = None
        self._execution_time = None
        self._memory_usage = None
        self._debug = debug

    def record_start_time(self):
        """Records the start time for performance measurement."""
        self._start_time = time.time()

    def record_end_time(self):
        """Records the end time and calculates the total execution time."""
        if self._start_time is None:
            raise RuntimeError("record_start_time() must be called before record_end_time()")
        self._execution_time = time.time() - self._start_time


    @abstractmethod
    def _mine(self):
        """
        The main method to run the algorithm's core logic.
        This method should be implemented by subclasses.
        """
        pass

    def mine(self):
        """
        Runs the algorithm and measures performance.
        If debug is True, it also runs memory and CPU profilers.
        """
        self.record_start_time()

        if self._debug:
            tracemalloc.start()
            profiler = cProfile.Profile()
            profiler.enable()

        self._mine()

        if self._debug:
            profiler.disable()
            profile_filename = f"{self.__class__.__name__}.prof"
            profiler.dump_stats(profile_filename)
            print(f"CPU profile saved to {profile_filename}. Run 'snakeviz {profile_filename}' to view.")

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._memory_usage = peak / 10**6  # Convert to MB
        elif platform.system() != "Windows":
            # Use a lightweight memory profiler for non-debug mode on Unix
            self._memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
        else:
            self._memory_usage = "Not measured on Windows (non-debug)"

        self.record_end_time()


    def get_execution_time(self):
        """Returns the execution time in seconds."""
        return self._execution_time

    def get_memory_usage(self):
        """Returns the peak memory usage in MB."""
        return self._memory_usage

    def get_pattern_count(self):
        """Returns the number of patterns found."""
        return len(self.patterns)

    @abstractmethod
    def save(self, oFile: Union[str, Path]):
        """
        Saves the found patterns to a file.
        This method must be implemented by subclasses.
        Args:
            oFile (str or Path): The path to the file where patterns should be saved.
        """
        pass

    def print_results(self):
        """Prints a summary of the algorithm's results."""
        print(f"--- {self.__class__.__name__} Results ---")
        print(f"Execution Time: {self.get_execution_time():.4f} seconds")
        print(f"Peak Memory Usage: {self.get_memory_usage():.2f} MB")
        print(f"Patterns Found: {self.get_pattern_count()}")
        print("--------------------" + "-" * len(self.__class__.__name__))
