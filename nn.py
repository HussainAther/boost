import numpy as np

from scipy.special import expit as activation_function
from scipy.stats import truncnorm

"""
Neural netowrk.
"""

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """
    Use the built-in scipy function to truncate.
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
