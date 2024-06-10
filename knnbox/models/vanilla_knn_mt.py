import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner


@register_model("vanilla_knn_mt")
class VanillaKNNMT(TransformerModel):
    r"""
    The vanilla knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
        parser.add_argument("--source-weight", type=float, metavar="D", default=0.0)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return VanillaKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class VanillaKNNMTDecoder(TransformerDecoder):
    r"""
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.cache_hidden = None
        self.encoder_hidden_len = 0
        self.current_gen_len = 0

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals", "keys", "hiddens", "hiddens_idx"])
            self.hidden_dic = []
            with open(os.path.join(args.knn_datastore_path, "dic.txt"), "r") as f:
                tmp = f.readlines()
                self.hidden_dic = [0] * len(tmp)
                for i in range(len(tmp)):
                    self.hidden_dic[i] = [int(x) for x in tmp[i].strip().split(" ")]

            self.hidden_dic = torch.tensor(self.hidden_dic).cuda()
            self.datastore.load_faiss_index("keys")
            self.datastore.load_faiss_index("hiddens")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))

            self.keys = 0
            self.ds_hiddens = 0
            self.gen_len = 1
            self.gen_end = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """

        self.gen_len = prev_output_tokens.shape[1]
        if self.args.knn_mode == "build_datastore":
            mask = self.datastore.get_pad_mask()
            encoder_hidden = encoder_out[0].transpose(0, 1)
            encoder_hidden = torch.mean(encoder_hidden, dim=1).unsqueeze(1)  # b x 1 x h

            non_pad_len = torch.sum(mask, dim=1)
            idx = torch.zeros(non_pad_len.sum(), dtype=int)
            non_pad_len = torch.cumsum(non_pad_len, dim=0)
            for i in range(len(non_pad_len)):
                if i != 0:
                    idx[non_pad_len[i - 1]: non_pad_len[i]] = self.encoder_hidden_len
                    self.encoder_hidden_len += 1
                else:
                    idx[:non_pad_len[i]] = self.encoder_hidden_len
                    self.encoder_hidden_len += 1

            assert idx[-1] - idx[0] == encoder_hidden.shape[0] - 1

            self.datastore["hiddens_idx"].add(idx)

            non_pad_hidden = encoder_hidden.squeeze(1)
            self.datastore["hiddens"].add(non_pad_hidden.half())

        encoder_hidden = encoder_out[0].transpose(0, 1)
        encoder_hidden = torch.mean(encoder_hidden, dim=1).unsqueeze(1)
        self.cache_hidden = encoder_hidden

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "build_datastore":
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference":
            self.retriever.retrieve(x, return_list=["keys", "vals", "distances", "hiddens_idx", "hiddens", "indices"])
        
        if not features_only:
            x = self.output_layer(x)
        return x, extra
    

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference":
            knn_prob = self.combiner.get_scaled_knn_prob(**self.retriever.results, device=net_output[0].device, cache_hidden=self.cache_hidden, current_gen_len=self.current_gen_len, source_weight=self.args.source_weight)

            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


r""" Define some vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
    

        

