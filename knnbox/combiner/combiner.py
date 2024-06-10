import sys

import torch
import torch.nn.functional as F

from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class Combiner:
    r"""
    A simple Combiner used by vanilla knn-mt
    """

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_scaled_knn_prob(self, vals, distances, temperature=None, device="cuda:0", cache_hidden=None, source_weight=0, **kwargs):
        temperature = temperature if temperature is not None else self.temperature
        if "hiddens" in kwargs:
            hiddens = kwargs["hiddens"]
            b, l, k, h = hiddens.shape
            retrieve_hidden = hiddens.float()
            cache_hidden = cache_hidden.float()
            retrieve_hidden = retrieve_hidden.view(b, -1, h)
            context_dist = torch.cdist(cache_hidden, retrieve_hidden, p=2)
            distances = distances + context_dist * source_weight
        scaled_dists = - distances / temperature

        knn_weights = torch.softmax(scaled_dists, dim=-1)
        B, S, K = vals.size()
        # construct prob
        knn_probs = torch.zeros(B, S, self.probability_dim, device=device)
        knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

        return knn_probs

    def get_knn_prob(self, vals, distances, temperature=None, device="cuda:0", **kwargs):
        r"""
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature  
        return calculate_knn_prob(vals, distances, self.probability_dim,
                     temperature, device, **kwargs)

    
    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_ = None, log_probs = False):
        r""" 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)
        

        