import torch
from knnbox.retriever.utils import retrieve_k_nearest

class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None
        self.sentence_retri_idx = None
        self.sentence_retri_hidden = None
        self.train_idx_dic = None
        self.train_hidden_dic = None
        self.source_hidden = None

    def retrieve(self, query, return_list=["vals", "distances"], k=None):
        r""" 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """
        k = k if k is not None else self.k
        if not hasattr(self.datastore, "faiss_index") or \
                self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)

        query = query.detach()
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], k)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        indices = faiss_results["indices"].cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query", "hiddens"]:
                assert data_name in self.datastore.datas, \
                    "You must load the {} of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        if "hiddens" in return_list:
            assert "hiddens_idx" in self.datastore.datas and "hiddens" in self.datastore.datas, \
                "You must load the hiddens of datastore first"
            ret["hiddens"] = torch.tensor(self.datastore["hiddens"].data[ret["hiddens_idx"].cpu().numpy()], device=query.device)

        self.results = ret  # save the retrieved results
        return ret

    @torch.no_grad()
    def fast_faiss_retrieve_with_hidden(self, query, hiddens, hidden_dic, gen_len, knn_mode, return_list=["vals", "distances"], k=None):
        ds_hiddens = self.datastore["hiddens"].data
        keys = self.datastore["keys"].data
        extra_hidden_vector = torch.full((1, keys.shape[1]), 999, device=query.device, dtype=torch.float)

        if knn_mode == "inference" and (gen_len == 1 or self.sentence_retri_hidden.shape[0] != query.shape[0]):
            b, s, h = hiddens.shape
            all_hiddens = ds_hiddens
            dic = hidden_dic
            faiss_hiddens = retrieve_k_nearest(hiddens, self.datastore.faiss_index["hiddens"], 1 * self.k)
            hiddens_idx = faiss_hiddens["indices"].cpu().numpy()
            retri_hiddens = torch.tensor(all_hiddens[hiddens_idx], device=query.device).float()
            hiddens_idx = torch.tensor(hiddens_idx, device=query.device).long()
            dis = torch.cdist(hiddens.unsqueeze(-2), retri_hiddens, p=2).squeeze(-2)

            k = real_k = self.k * 1
            sorted_dis, sorted_idx = torch.sort(dis, dim=-1)
            sorted_idx = sorted_idx[:, :, :real_k]
            real_hiddens_idx = hiddens_idx.gather(2, sorted_idx)
            sorted_idx = real_hiddens_idx

            dic_start = dic[sorted_idx[:, :, :real_k], 0]
            dic_end = dic[sorted_idx[:, :, :real_k], 1]
            dic_end = torch.where(dic_end - dic_start < 50, dic_end, dic_start + 50) # avoid too big datastore
            max_len = (dic_end - dic_start).max().item()

            max_key_idx = torch.tensor(keys.shape[0] - 1, device=query.device)
            tmp_arange = torch.arange(max_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, s, k, -1).to(query.device)
            tmp_arange = tmp_arange + dic_start.unsqueeze(-1)
            tmp_arange = torch.where(tmp_arange < dic_end.unsqueeze(-1), tmp_arange, max_key_idx)
            target_idx = tmp_arange.reshape(b, s, -1)
            target_idx, _ = torch.sort(target_idx, dim=-1)
            target_mask = target_idx != max_key_idx
            new_len = torch.max(target_mask.long().sum(dim=-1))
            target_idx = target_idx[:, :, :new_len]
            self.sentence_retri_idx = target_idx

            valid_mask = target_idx != max_key_idx
            target_idx = target_idx * valid_mask
            target_hidden = torch.tensor(keys[target_idx.cpu().numpy()], device=query.device).float()
            new_extra_hidden_vector = extra_hidden_vector.unsqueeze(0).unsqueeze(0).expand(*target_hidden.shape)
            target_hidden = torch.where(valid_mask.unsqueeze(-1).expand(-1, -1, -1, new_extra_hidden_vector.shape[-1]), target_hidden, new_extra_hidden_vector)
            self.sentence_retri_hidden = target_hidden

        if knn_mode == "inference":
            dis = torch.cdist(query.unsqueeze(-2), self.sentence_retri_hidden, p=2).squeeze(-2)
            dis, idx = torch.sort(dis, dim=-1)
            idx = idx[:, :, :self.k * 1]
            batch_idx = self.sentence_retri_idx.gather(2, idx)
            batch_hiddens = self.sentence_retri_hidden.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, self.sentence_retri_hidden.shape[-1]))

        if knn_mode == "train_metak":
            b, s, h = query.shape
            all_hiddens = ds_hiddens
            dic = hidden_dic
            hiddens = hiddens.expand(-1, s, -1)
            faiss_hiddens = retrieve_k_nearest(hiddens, self.datastore.faiss_index["hiddens"], 1 * self.k)
            hiddens_idx = faiss_hiddens["indices"].cpu().numpy()
            retri_hiddens = torch.tensor(all_hiddens[hiddens_idx], device=query.device).float()
            hiddens_idx = torch.tensor(hiddens_idx, device=query.device).long()
            dis = torch.cdist(hiddens.unsqueeze(-2), retri_hiddens, p=2).squeeze(-2)

            k = real_k = self.k * 1
            sorted_dis, sorted_idx = torch.sort(dis, dim=-1)
            sorted_idx = sorted_idx[:, :, :real_k]
            real_hiddens_idx = hiddens_idx.gather(2, sorted_idx)
            sorted_idx = real_hiddens_idx

            dic_start = dic[sorted_idx[:, :, :real_k], 0]
            dic_end = dic[sorted_idx[:, :, :real_k], 1]
            dic_end = torch.where(dic_end - dic_start < 50, dic_end, dic_start + 50) # avoid too big datastore
            max_len = (dic_end - dic_start).max().item()

            max_key_idx = torch.tensor(keys.shape[0] - 1, device=query.device)
            tmp_arange = torch.arange(max_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, s, k, -1).to(query.device)
            tmp_arange = tmp_arange + dic_start.unsqueeze(-1)
            tmp_arange = torch.where(tmp_arange < dic_end.unsqueeze(-1), tmp_arange, max_key_idx)
            target_idx = tmp_arange.reshape(b, s, -1)
            target_idx, _ = torch.sort(target_idx, dim=-1)
            target_mask = target_idx != max_key_idx
            new_len = torch.max(target_mask.long().sum(dim=-1))
            target_idx = target_idx[:, :, :new_len]

            valid_mask = target_idx != max_key_idx
            target_idx = target_idx * valid_mask
            target_hidden = torch.tensor(keys[target_idx.cpu().numpy()], device=query.device).float()
            new_extra_hidden_vector = extra_hidden_vector.unsqueeze(0).unsqueeze(0).expand(*target_hidden.shape)
            target_hidden = torch.where(valid_mask.unsqueeze(-1).expand(-1, -1, -1, new_extra_hidden_vector.shape[-1]), target_hidden, new_extra_hidden_vector)

            dis = torch.cdist(query.unsqueeze(-2), target_hidden, p=2).squeeze(-2)
            dis, idx = torch.sort(dis, dim=-1)
            idx = idx[:, :, :self.k * 1]
            batch_idx = target_idx.gather(2, idx)
            batch_hiddens = target_hidden.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, target_hidden.shape[-1]))

            torch.cuda.empty_cache()

        k = k if k is not None else self.k
        if not hasattr(self.datastore, "faiss_index") or \
                self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)

        query = query.detach()
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], k)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
            ori_indices = ret["indices"].clone()
            ret["indices"] = torch.cat([ret["indices"], batch_idx.to(query.device)], dim=-1)
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        ori_indices = ori_indices.cpu().numpy()
        indices = ret["indices"].cpu().numpy()

        if "vals" in return_list:
            ret["vals"] = torch.tensor(self.datastore["vals"].data[indices], device=query.device)
        if "hiddens_idx" in return_list:
            ret["hiddens_idx"] = torch.tensor(self.datastore["hiddens_idx"].data[indices], device=query.device)
        if "hiddens" in return_list:
            ret["hiddens"] = torch.tensor(self.datastore["hiddens"].data[ret["hiddens_idx"].cpu().numpy()], device=query.device)
        if "keys" in return_list:
            ori_keys = torch.tensor(self.datastore["keys"].data[ori_indices], device=query.device)
            ret["keys"] = torch.cat([ori_keys, batch_hiddens], dim=-2)
        if "distances" in return_list:
            ret["distances"] = torch.cdist(query.unsqueeze(-2), ret["keys"].float(), p=2).squeeze(-2) # calculate the distance between query and retrieved keys
        if "query" in return_list:
            ret["query"] = query

        self.results = ret
        return ret
