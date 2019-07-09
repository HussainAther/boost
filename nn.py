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

class NeuralNetwork:
           
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
