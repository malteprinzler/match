import argparse
import os
import re

import torch


def _find_state_dict_key(ckpt):
    for candidate in ("model", "model_state", "state_dict"):
        value = ckpt.get(candidate, None)
        if isinstance(value, dict):
            return candidate
    raise KeyError(
        "Could not find a state dict in checkpoint. "
        "Expected one of: 'model', 'model_state', 'state_dict'."
    )


def map_inofficial_to_match_key(key):
    if key.startswith("_feature_net."):
        rest = key[len("_feature_net.") :]
        rest = rest.replace("_stem_conv.", "model.input_conv.")
        rest = rest.replace("_stem_pool.", "model.input_pool.")
        rest = re.sub(r"_enc_block(\d+)\.(\d+)\.", lambda m: f"model.enc{m.group(1)}.block{int(m.group(2)) + 1}.", rest)
        rest = re.sub(r"_decoder_block(\d+)\.", r"model.dec\1.", rest)
        rest = re.sub(r"_skip(\d+)\.", r"model.skip\1.", rest)
        rest = rest.replace("_final_layer.", "model.out.")
        rest = rest.replace("._up_conv.", ".up_conv.")
        rest = rest.replace("._conv1.", ".conv1.")
        rest = rest.replace("._conv2.", ".conv2.")
        rest = rest.replace("._skip.", ".skip.")
        rest = rest.replace("._conv.", ".conv.")
        rest = re.sub(r"\._([A-Za-z])", r".\1", rest)
        return f"feature_net.{rest}"

    if key.startswith("_global_stage_net._prediction_net."):
        rest = key[len("_global_stage_net._prediction_net.") :]
        rest = rest.replace("_front_layers.", "front_layers.")
        rest = rest.replace("_encoder_decoder.", "encoder_decoder.")
        rest = rest.replace("_back_layers.", "back_layers.")
        rest = rest.replace("_out_layer", "output_layer")
        rest = rest.replace("._block.", ".block.")
        rest = rest.replace("._up_conv.", ".block.0.")
        rest = rest.replace("._conv1.", ".res_branch.0.")
        rest = rest.replace("._conv2.", ".res_branch.3.")
        rest = rest.replace("._skip.", ".skip_con.0.")
        rest = re.sub(r"\._([A-Za-z])", r".\1", rest)
        return f"sparse_point_net.global_net.{rest}"

    return None


def convert_checkpoint(
    inofficial_ckpt_path,
    official_ckpt_path,
    output_ckpt_path,
    map_location="cpu",
    strict=False,
):
    inofficial_ckpt = torch.load(inofficial_ckpt_path, map_location=map_location)
    official_ckpt = torch.load(official_ckpt_path, map_location=map_location)

    inofficial_sd_key = _find_state_dict_key(inofficial_ckpt)
    official_sd_key = _find_state_dict_key(official_ckpt)

    inofficial_sd = inofficial_ckpt[inofficial_sd_key]
    official_sd = official_ckpt[official_sd_key]

    copied = 0
    unmapped = []
    missing_in_target = []
    shape_mismatch = []

    for src_key, src_tensor in inofficial_sd.items():
        dst_key = map_inofficial_to_match_key(src_key)
        if dst_key is None:
            unmapped.append(src_key)
            continue

        if dst_key not in official_sd:
            missing_in_target.append((src_key, dst_key))
            continue

        dst_tensor = official_sd[dst_key]
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            shape_mismatch.append((src_key, dst_key, tuple(src_tensor.shape), tuple(dst_tensor.shape)))
            continue

        official_sd[dst_key] = src_tensor
        copied += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_ckpt_path)), exist_ok=True)
    official_ckpt[official_sd_key] = official_sd
    torch.save(official_ckpt, output_ckpt_path)

    print(f"Inofficial state dict key: {inofficial_sd_key}")
    print(f"Official state dict key:   {official_sd_key}")
    print(f"Copied tensors:            {copied}")
    print(f"Unmapped source keys:      {len(unmapped)}")
    print(f"Missing target keys:       {len(missing_in_target)}")
    print(f"Shape mismatches:          {len(shape_mismatch)}")
    print(f"Saved converted checkpoint: {output_ckpt_path}")

    if unmapped:
        print("\nFirst unmapped source keys:")
        for key in unmapped[:10]:
            print(f"  - {key}")
    if missing_in_target:
        print("\nFirst missing target mappings:")
        for src_key, dst_key in missing_in_target[:10]:
            print(f"  - {src_key} -> {dst_key}")
    if shape_mismatch:
        print("\nFirst shape mismatches:")
        for src_key, dst_key, src_shape, dst_shape in shape_mismatch[:10]:
            print(f"  - {src_key} -> {dst_key}: {src_shape} vs {dst_shape}")

    if strict and (unmapped or missing_in_target or shape_mismatch):
        raise RuntimeError("Conversion incomplete in --strict mode.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy network weights from an inofficial TEMPEH-style checkpoint "
            "into an official MATCH checkpoint."
        )
    )
    parser.add_argument("--inofficial-ckpt", required=True, help="Path to source inofficial checkpoint.")
    parser.add_argument("--official-ckpt", required=True, help="Path to target/template official checkpoint.")
    parser.add_argument("--output-ckpt", required=True, help="Path where converted checkpoint will be saved.")
    parser.add_argument("--map-location", default="cpu", help="torch.load map_location (default: cpu).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any key is unmapped, missing in target, or shape-mismatched.",
    )
    return parser.parse_args()


"""
python scripts/tempeh/inofficial_to_match_ckpts.py \
    --inofficial-ckpt /is/cluster/mprinzler/gtempeh/experiments/tempeh/finalava256/finalava256_tempeh/train/checkpoints/01000000.ckpt \
    --official-ckpt /is/cluster/mprinzler/projects/gintern/match/experiments/tempeh/closedgap/coarse__tempeh_closedgap__March05__15-12-04/checkpoints/model_01860000.pth \
    --output-ckpt /is/cluster/mprinzler/projects/gintern/match/experiments/tempeh/closedgap/coarse__tempeh_closedgap__March05__15-12-04/checkpoints/model_01860000_inofficial.pth
"""
if __name__ == "__main__":
    args = parse_args()
    convert_checkpoint(
        inofficial_ckpt_path=args.inofficial_ckpt,
        official_ckpt_path=args.official_ckpt,
        output_ckpt_path=args.output_ckpt,
        map_location=args.map_location,
        strict=args.strict,
    )