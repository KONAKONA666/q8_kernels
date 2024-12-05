import torch
import os
import argparse

from safetensors.torch import load_file
from pathlib import Path
from q8_kernels.functional.quantizer import quantize
from q8_kernels.functional.fast_hadamard import hadamard_transform

from q8_kernels.models.LTXVideo import get_ltx_transformer

def load_unet(unet_path: Path):
    if os.path.isdir(str(unet_path)):
        unet_path = unet_path / "unet_diffusion_pytorch_model.safetensors"
    
    assert str(unet_path.name.endswith(".safetensors")), f"{unet_path} is not proper file"
    assert unet_path.exists(), f"{unet_path} no in existancce"

    unet_state_dict = load_file(str(unet_path))

    return unet_state_dict

def convert_state_dict(orig_state_dict):
    prefix = "transformer_blocks"
    to_fp32 = ["norm", "bias"]
    transformer_block_keys = []
    non_transformer_block_keys = []
    for k in orig_state_dict:
        if prefix in k:
            transformer_block_keys.append(k)
        else:
            non_transformer_block_keys.append(k)
    attn_keys = []
    ffn_keys = []
    scale_shift_keys = []
    for k in transformer_block_keys:
        if "attn" in k:
            attn_keys.append(k)
    for k in transformer_block_keys:
        if "ff" in k:
            ffn_keys.append(k)
    for k in transformer_block_keys:
        if "scale_shift_table" in k:
            scale_shift_keys.append(k)

    assert len(attn_keys + ffn_keys + scale_shift_keys) == len(transformer_block_keys), "error"
    
    new_state_dict = {}
    for k in attn_keys:
        new_key = k
        new_key = new_key.replace("to_out.0.", "to_out.")
        if "norm" in k and "weight" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in ffn_keys:
        new_key = k.replace(".net.0.", ".act.")
        new_key = new_key.replace(".net.2.", ".proj_down.")
        
        if "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in scale_shift_keys:
        new_state_dict[k] = orig_state_dict[k]

    for k in non_transformer_block_keys:
        new_state_dict[k] = orig_state_dict[k]
    
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    
    args = parser.parse_args()
    
    orig_state_dict = load_unet(args.input_path)
    new_state_dict = convert_state_dict(orig_state_dict)

    unet = get_ltx_transformer().cuda()

    m, u = unet.load_state_dict(new_state_dict, strict=True)
    
    unet.save_pretrained(args.output_path)

    print(f"model saved in {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    
    args = parser.parse_args()
    
    main(args)