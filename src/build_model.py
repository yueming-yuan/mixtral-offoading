import os
import json
from functools import cache
from dataclasses import dataclass
import typing as tp

import torch
from torch import nn

from transformers import AutoConfig
from transformers.models.mixtral import MixtralForCausalLM, MixtralConfig

from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange

from hqq.core.quantize import BaseQuantizeConfig

from .expert_cache import ExpertCache, ExpertCache_nq
from .expert_wrapper import MixtralExpertWrapper, MixtralExpertWrapper_nq
from .custom_layers import (
    HQQLinearTritonSavable,
    MixtralBLockSparseTop2MLP_HQQ,
    SparseMoeWrapper,
)
from .utils import with_default_dtype

import gc
import re

from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP

@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int


class QuantConfig:
    def __init__(
        self,
        ffn_config: BaseQuantizeConfig,
        attn_config: BaseQuantizeConfig,
    ):
        self.ffn_config = ffn_config
        self.attn_config = attn_config

    @cache
    def get_ffn_metas(self, hidden_dim: int, ffn_dim: int) -> tuple[tp.Any, tp.Any]:
        return (
            HQQLinearTritonSavable.get_hqq_meta((hidden_dim, ffn_dim), self.ffn_config),
            HQQLinearTritonSavable.get_hqq_meta((ffn_dim, hidden_dim), self.ffn_config),
        )


def replace_attn_layers(
    model: MixtralForCausalLM,
    config: MixtralConfig,
    quant_config: QuantConfig,
    device: torch.device,
) -> None:
    attn_quant_config = quant_config.attn_config

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]

    shape_to_meta = {
        shape: HQQLinearTritonSavable.get_hqq_meta(shape, attn_quant_config)
        for shape in shapes
    }

    def patch_fct_hqq(shape, quant_config):
        meta = shape_to_meta[shape]
        layer = HQQLinearTritonSavable(None, quant_config, meta=meta)
        return layer

    for layer in model.model.layers:
        layer.block_sparse_moe.gate = nn.Linear(
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )

        layer.self_attn.q_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_quant_config
        )
        layer.self_attn.k_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_quant_config
        )
        layer.self_attn.v_proj = patch_fct_hqq(
            (hidden_size, num_key_value_heads * head_dim), attn_quant_config
        )
        layer.self_attn.o_proj = patch_fct_hqq(
            (hidden_size, num_heads * head_dim), attn_quant_config
        )


def replace_attn_layers_nq(
    model: MixtralForCausalLM,
    config: MixtralConfig,
    device: torch.device,
) -> None:

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_key_value_heads = config.num_key_value_heads

    shapes = [
        (hidden_size, num_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (hidden_size, num_key_value_heads * head_dim),
        (num_heads * head_dim, hidden_size),
    ]
    for layer in model.model.layers:
        layer.block_sparse_moe.gate = nn.Linear(
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.float16,
            device=device,
            bias=False,
        )

        # layer.self_attn.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=torch.float16,device=device)
        # layer.self_attn.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=torch.float16,device=device)
        # layer.self_attn.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, dtype=torch.float16,device=device)
        # layer.self_attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=torch.float16,device=device)


@cache
def get_default_ffn_quant_config(ffn_dim: int = 14336, hidden_dim: int = 4096):
    quant_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )

    meta1 = HQQLinearTritonSavable.get_hqq_meta((hidden_dim, ffn_dim), quant_config)
    meta2 = HQQLinearTritonSavable.get_hqq_meta((ffn_dim, hidden_dim), quant_config)

    return quant_config, meta1, meta2


def make_empty_expert(
    model_config: MixtralConfig, quant_config: QuantConfig
) -> MixtralBLockSparseTop2MLP_HQQ:
    meta1, meta2 = quant_config.get_ffn_metas(
        model_config.hidden_size, model_config.intermediate_size
    )
    return MixtralBLockSparseTop2MLP_HQQ(
        model_config,
        quant_config.ffn_config,
        meta1,
        meta2,
    )


def make_and_load_expert_wrapper(
    config: MixtralConfig,
    quant_config: QuantConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]

    state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = make_empty_expert(config, quant_config)
    expert.load_state_dict(state_dict, strict=True)

    return MixtralExpertWrapper(expert, device)


def load_00_expert_state_dict(states_dir: str, device: torch.device):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.0.block_sparse_moe.experts.0"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.W_q"]
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))


def build_model(
    device: torch.device,
    quant_config: QuantConfig,
    offload_config: OffloadConfig,
    state_path: str,
):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    state_dict_00 = load_00_expert_state_dict(state_path, device)

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config, quant_config)
        expert.load_state_dict(state_dict_00)
        return MixtralExpertWrapper(expert, device=device)

    with device, with_default_dtype(torch.float16):
        model = MixtralForCausalLM(
            AutoConfig.from_pretrained(
                model_name,
                num_local_experts=0,
                torch_dtype=torch.float16,
                device_map=device,
            ),
        )

    model_config = AutoConfig.from_pretrained(model_name)
    replace_attn_layers(model, model_config, quant_config, device)
    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    trunk_state_path = os.path.join(
        state_path,
        weight_map["model.embed_tokens.weight"],
    )  
    print(load_file(trunk_state_path, device=str(device)).keys())
    model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=True)

    expert_cache = ExpertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
    )
    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = SparseMoeWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            expert_wrapper = make_and_load_expert_wrapper(
                config=model_config,
                quant_config=quant_config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device=device,
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return model

def load_00_expert_state_dict_nq(states_dir: str, device: torch.device):
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.0.block_sparse_moe.experts.0"
        state_fpath = json.load(f)["weight_map"][f"{module_idx}.w1.weight"]
    return load_file(os.path.join(states_dir, state_fpath), device=str(device))

def make_and_load_expert_wrapper_nq(
    config: MixtralConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper_nq:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        weight_map = json.load(f)["weight_map"]
        state_fpath = weight_map[f"{module_idx}.w1.weight"]
    loaded_state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    expert = MixtralBLockSparseTop2MLP(config).half()
    expert.load_state_dict(loaded_state_dict, strict=True)

    """
    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        weight_map = json.load(f)["weight_map"]
        state_fpath = weight_map[f"{module_idx}.w1.weight"]
        state_fpath2 = weight_map[f"{module_idx}.w3.weight"]
    
    loaded_state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    state_dict = {}
    state_dict["w1.weight"] = loaded_state_dict[f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight'] 

    if f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight' not in loaded_state_dict:
        loaded_state_dict = load_file(os.path.join(states_dir, state_fpath2), device=str(device))
    state_dict["w2.weight"] = loaded_state_dict[f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight'] 
    
    if f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight' not in loaded_state_dict:
        loaded_state_dict = load_file(os.path.join(states_dir, state_fpath2), device=str(device))
    state_dict["w3.weight"] = loaded_state_dict[f'model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight'] 

    del loaded_state_dict

    # state_dict = load_file(os.path.join(states_dir, state_fpath), device=str(device))
    # expert = make_empty_expert(config, quant_config)
    expert = MixtralBLockSparseTop2MLP(config).half()
    #print(expert.state_dict)
    #print(state_dict)
    expert.load_state_dict(state_dict, strict=True)
    """

    return MixtralExpertWrapper_nq(expert, device)


def build_model_without_quant(
    device: torch.device,
    offload_config: OffloadConfig,
    state_path: str,
):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    state_dict_00 = load_00_expert_state_dict_nq(state_path, device)
    """
    state_dict_00 = {}
    key_mapping = {'model.layers.0.block_sparse_moe.experts.0.w1.weight': 'w1.weight', 'model.layers.0.block_sparse_moe.experts.0.w2.weight': 'w2.weight', 'model.layers.0.block_sparse_moe.experts.0.w3.weight': 'w3.weight'}
    for i, j in key_mapping.items():
        state_dict_00[j] = loaded_state_dict_00[i]
    del loaded_state_dict_00
    """

    def _make_module():
        config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
        # expert = make_empty_expert(config, quant_config)
        expert = MixtralBLockSparseTop2MLP(config).half()
        expert.load_state_dict(state_dict_00)
        return MixtralExpertWrapper_nq(expert, device=device)

    with device, with_default_dtype(torch.float16):
        model = MixtralForCausalLM(
            AutoConfig.from_pretrained(
                model_name,
                num_local_experts=0,
                torch_dtype=torch.float16,
                device_map=device,
            ),
        )

    model_config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
    replace_attn_layers_nq(model, model_config, device)
    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]

    '''
    state_dict = {}
    exclusion_pattern = re.compile(r"model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.(w1|w2|w3)\.weight")
    unique_filenames = set(weight_map.values())
    for filename in unique_filenames:
        file_path = os.path.join(state_path, filename)
        loaded_file = load_file(file_path, device=str(device))
        for key, file_in_map in weight_map.items():
            if filename == file_in_map and not exclusion_pattern.search(key):
                if key in loaded_file:
                    # print(f"Adding tensor to state_dict: {key}")
                    state_dict[key] = loaded_file[key]
                else:
                    print(f"Expected tensor not found in safetensor file: {key}")
        peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) 
        print(f"Peak GPU Memory Usage: {peak_memory_usage} GB")  
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)
    '''

    trunk_state_path = os.path.join(
        state_path,
        weight_map["model.embed_tokens.weight"],
    )
    print(trunk_state_path)
    print(load_file(trunk_state_path, device=str(device)))
    model.load_state_dict(load_file(trunk_state_path, device=str(device)), strict=True)

    print(f"Peak GPU Memory Usage: {peak_memory_usage} GB")  
    print("Created expert cache! ... Cleaning cache")
    print("Finish loading trunk states!")

    expert_cache = ExpertCache_nq(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
    )
    peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"Peak GPU Memory Usage: {peak_memory_usage} GB")  
    print("Created expert cache! ... Cleaning cache")
    del state_dict_00
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleaned.")
    peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"Peak GPU Memory Usage: {peak_memory_usage} GB")  

    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = SparseMoeWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            expert_wrapper = make_and_load_expert_wrapper_nq(
                config=model_config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device=device,
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return model