"""
Utility functions for Creative Fatigue Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def ensure_dir(path: str):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def cyclical_encode(series: pd.Series, max_val: int) -> Tuple[pd.Series, pd.Series]:
    """
    Encode cyclical features (hour, day of week, etc.) using sin/cos.
    
    Parameters:
    -----------
    series : pd.Series
        Series with cyclical values (0 to max_val)
    max_val : int
        Maximum value in the cycle (e.g., 23 for hours, 6 for days)
        
    Returns:
    --------
    Tuple of (sin_encoded, cos_encoded) series
    """
    sin_encoded = np.sin(2 * np.pi * series / max_val)
    cos_encoded = np.cos(2 * np.pi * series / max_val)
    return sin_encoded, cos_encoded

