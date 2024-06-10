import torch
from torch import nn
from knnbox.combiner.utils import calculate_combined_prob, calculate_knn_prob


class MuCalculator(nn.Module):

    def __init__(self,
                 max_k,
                 use_context_dis,
                 **kwargs
                 ):
        super().__init__()
        self.use_context_dis = use_context_dis
        self.meta_k_network = MetaKNetwork(max_k, use_context_dis=self.use_context_dis, **kwargs)
        self.max_k = max_k
        self.kwargs = kwargs
        self.mask_for_distance = None
        self.mu = None

    def robust_calculate_new_distance(self, vals, distances, device="cuda:0", cache_hidden=None, hiddens=None, **kwargs):
        b, l, k, h = hiddens.shape
        retrieve_hidden = hiddens.float()
        cache_hidden = cache_hidden.float()
        retrieve_hidden = retrieve_hidden.view(b, -1, h)
        context_dist = torch.cdist(cache_hidden, retrieve_hidden, p=2)
        context_dist = context_dist.view(b, l, -1)

        scale = self.meta_k_network(vals, distances, context_dist)["mu_net_output"] * 5 # scale the context_dist
        context_dist = context_dist * scale
        distances = distances + context_dist

        return context_dist, distances

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" get combined probs of knn_prob and neural_model_prob """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)


class MetaKNetwork(nn.Module):
    def __init__(
            self,
            max_k=32,
            use_context_dis=False,
            lambda_net_hid_size=32,
            label_count_as_feature=True,
            relative_label_count=False,
            lambda_net_dropout_rate=0.0,
            device="cuda:0",
            **kwargs,
    ):
        super().__init__()
        self.max_k = max_k
        self.use_context_dis = use_context_dis
        self.label_count_as_feature = label_count_as_feature
        self.relative_label_count = relative_label_count
        self.device = device
        self.mask_for_label_count = None

        if self.use_context_dis is True:
            self.distance_to_mu = nn.Sequential(
                nn.Linear(self.max_k * 3 if self.label_count_as_feature else self.max_k, lambda_net_hid_size * 2),
                nn.Tanh(),
                nn.Dropout(p=lambda_net_dropout_rate),
                nn.Linear(lambda_net_hid_size * 2, 1),
                nn.Sigmoid()
            )

            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.distance_to_mu[0].weight[:, :self.max_k], gain=1)
                nn.init.xavier_normal_(self.distance_to_mu[0].weight[:, self.max_k: 2 * self.max_k], gain=1)
                nn.init.xavier_normal_(self.distance_to_mu[0].weight[:, 2 * self.max_k:], gain=1)
            else:
                nn.init.normal_(self.distance_to_mu[0].weight, mean=0, std=0.01)

        elif self.use_context_dis is False:
            self.distance_to_mu = nn.Sequential(
                nn.Linear(self.max_k * 2 if self.label_count_as_feature else self.max_k, lambda_net_hid_size * 4),
                nn.Tanh(),
                nn.Dropout(p=lambda_net_dropout_rate),
                nn.Linear(lambda_net_hid_size * 4, 1),
                nn.Sigmoid()
            )

            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.distance_to_mu[0].weight[:, : self.max_k], gain=0.01)
                nn.init.xavier_normal_(self.distance_to_mu[0].weight[:, self.max_k: 2 * self.max_k], gain=0.1)
                nn.init.xavier_normal_(self.distance_to_mu[-2].weight)
            else:
                nn.init.normal_(self.distance_to_mu[0].weight, mean=0, std=0.01)

    def forward(self, vals, distances, encoder_distances):
        if self.label_count_as_feature:
            label_counts = self._get_label_count_segment(vals, relative=self.relative_label_count)
            if self.use_context_dis is True:
                network_inputs = torch.cat((encoder_distances.detach(), distances.detach(), label_counts.detach().float()), dim=-1)
            elif self.use_context_dis is False:
                network_inputs = torch.cat((distances.detach(), label_counts.detach().float()), dim=-1)
        else:
            network_inputs = distances.detach()

        results = {}

        results["mu_net_output"] = self.distance_to_mu(network_inputs)

        return results

    def _get_label_count_segment(self, vals, relative=False):
        r""" this function return the label counts for different range of k nearest neighbor
            [[0:0], [0:1], [0:2], ..., ]
        """

        # caculate `label_count_mask` only once
        if self.mask_for_label_count is None:
            mask_for_label_count = torch.empty((self.max_k, self.max_k)).fill_(1)
            mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
            mask_for_label_count.requires_grad = False
            # [0,1,1]
            # [0,0,1]
            # [0,0,0]
            self.mask_for_label_count = mask_for_label_count.to(vals.device)

        ## TODO: The feature below may be unreasonable
        B, S, K = vals.size()
        expand_vals = vals.unsqueeze(-2).expand(B, S, K, K)
        expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)

        labels_sorted, _ = expand_vals.sort(dim=-1)  # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        retrieve_label_counts[:, :, :-1] -= 1

        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]

        return retrieve_label_counts
