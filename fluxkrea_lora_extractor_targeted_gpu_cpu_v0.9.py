# extract approximating LoRA by svd from two FLUX models
# The code is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Updated to allow GPU/CPU/auto device selection and safer fallback from GPU OOM.

import argparse
import json
import os
import time
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
# --- Add possible repo paths so local `library` can be imported ---------------------
import os, sys

_this_dir = os.path.dirname(os.path.abspath(__file__))      # e.g. .../kohya_ss/sd-scripts/networks
# candidate locations where the `library` package might live:
candidates = [
    os.path.abspath(os.path.join(_this_dir, '..', '..')),  # .../kohya_ss
    os.path.abspath(os.path.join(_this_dir, '..')),        # .../kohya_ss/sd-scripts
    os.path.abspath(os.path.join(_this_dir, '..', '..', '..')),  # one level higher just in case
]

# Insert the first candidate that exists (and isn't already in sys.path)
for p in candidates:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# (Optional) helpful debug print when running interactively:
if os.getenv("FLUX_VERBOSE_SYS_PATH", None):
    print("Inserted paths into sys.path for local imports:", candidates)
# ----------------------------------------------------------------------------------


from library import flux_utils, sai_model_spec, model_util, sdxl_model_util
import lora
from library.utils import MemoryEfficientSafeOpen
from library.utils import setup_logging
from networks import lora_flux

setup_logging()
import logging

logger = logging.getLogger(__name__)


def save_to_file(file_name, state_dict, metadata, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    save_file(state_dict, file_name, metadata=metadata)


def should_skip_key(key, args):
    """
    Determine if a key should be skipped based on filtering options
    """
    # Skip double blocks if specified
    if args.skip_double_blocks and "double_block" in key:
        return True
    
    # Skip single blocks if specified
    if args.skip_single_blocks and "single_block" in key:
        return True
    
    # Skip KQV attention keys if specified
    if args.skip_kqv and any(kqv in key for kqv in ["to_k", "to_q", "to_v"]):
        return True
    
    # Skip projection attention keys if specified
    if args.skip_proj and "to_out" in key:
        return True
    
    # Skip MLP 0 keys if specified
    if args.skip_mlp0 and "mlp.0" in key:
        return True
    
    # Skip MLP 2 keys if specified
    if args.skip_mlp2 and "mlp.2" in key:
        return True
    
    # Skip txt-based keys if specified
    if args.skip_txt and "txt" in key:
        return True
    
    # Skip img-based keys if specified
    if args.skip_img and "img" in key:
        return True
    
    # Skip linear1 keys if specified
    if args.skip_linear1 and "linear1" in key:
        return True
    
    # Skip linear2 keys if specified
    if args.skip_linear2 and "linear2" in key:
        return True
    
    # Skip modulation keys if specified
    if args.skip_modulation and "modulation" in key:
        return True
    
    return False


def svd(
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    device=None,
    save_precision=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
    mem_eff_safe_open=False,
    skip_double_blocks=False,
    skip_single_blocks=False,
    skip_kqv=False,
    skip_proj=False,
    skip_mlp0=False,
    skip_mlp2=False,
    skip_txt=False,
    skip_img=False,
    skip_linear1=False,
    skip_linear2=False,
    skip_modulation=False,
):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    calc_dtype = torch.float
    save_dtype = str_to_dtype(save_precision)
    store_device = "cpu"

    # Create args object for skip checking
    class Args:
        def __init__(self):
            self.skip_double_blocks = skip_double_blocks
            self.skip_single_blocks = skip_single_blocks
            self.skip_kqv = skip_kqv
            self.skip_proj = skip_proj
            self.skip_mlp0 = skip_mlp0
            self.skip_mlp2 = skip_mlp2
            self.skip_txt = skip_txt
            self.skip_img = skip_img
            self.skip_linear1 = skip_linear1
            self.skip_linear2 = skip_linear2
            self.skip_modulation = skip_modulation
    
    args = Args()

    # resolve device choice
    def resolve_device(device_str: str):
        # device_str may be None, 'cpu', 'cuda', or 'auto'
        if device_str is None or device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("Requested --device cuda but CUDA is not available; falling back to cpu")
                return torch.device("cpu")
        if device_str == "cpu":
            return torch.device("cpu")
        # fallback
        return torch.device(device_str)

    device_obj = resolve_device(device)
    logger.info(f"Using device for computation: {device_obj}")

    # open models
    lora_weights = {}
    if not mem_eff_safe_open:
        # use original safetensors.safe_open
        open_fn = lambda fn: safe_open(fn, framework="pt")
    else:
        logger.info("Using memory efficient safe_open")
        open_fn = lambda fn: MemoryEfficientSafeOpen(fn)

    with open_fn(model_org) as f_org:
        # filter keys
        keys = []
        skipped_count = 0
        for key in f_org.keys():
            if not ("single_block" in key or "double_block" in key):
                continue
            if ".bias" in key:
                continue
            if "norm" in key:
                continue
            
            # Check if key should be skipped based on filtering options
            if should_skip_key(key, args):
                skipped_count += 1
                continue
            
            keys.append(key)
        
        logger.info(f"Processing {len(keys)} keys, skipped {skipped_count} keys based on filtering options")

        with open_fn(model_tuned) as f_tuned:
            for key in tqdm(keys):
                # get tensors and calculate difference
                value_o = f_org.get_tensor(key)
                value_t = f_tuned.get_tensor(key)

                # move to calc dtype (stay on CPU first to avoid duplicating memory on GPU)
                mat = value_t.to(calc_dtype) - value_o.to(calc_dtype)
                del value_o, value_t

                # attempt to move mat to the requested device for SVD
                if device_obj.type == "cuda":
                    try:
                        mat = mat.to(device_obj)
                    except RuntimeError as e:
                        logger.warning(f"Failed to move tensor to CUDA for key {key}: {e}; will run SVD on CPU")
                        device_for_svd = torch.device("cpu")
                    else:
                        device_for_svd = device_obj
                else:
                    device_for_svd = device_obj

                out_dim, in_dim = mat.size()[0:2]
                rank = min(dim, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

                mat = mat.squeeze()

                # run SVD, with fallback to CPU if GPU throws OOM or other error
                try:
                    if device_for_svd.type == "cuda":
                        U, S, Vh = torch.linalg.svd(mat)
                    else:
                        # Ensure mat is on CPU
                        if mat.device.type != "cpu":
                            mat = mat.to("cpu")
                        U, S, Vh = torch.linalg.svd(mat)
                except RuntimeError as e:
                    logger.warning(f"SVD failed on device {device_for_svd} for key {key}: {e}; retrying on CPU")
                    if mat.device.type != "cpu":
                        mat = mat.to("cpu")
                    torch.cuda.empty_cache()
                    U, S, Vh = torch.linalg.svd(mat)

                # reduce rank and form LoRA factors
                U = U[:, :rank]
                S = S[:rank]
                U = U @ torch.diag(S)

                Vh = Vh[:rank, :]

                dist = torch.cat([U.flatten(), Vh.flatten()])
                hi_val = torch.quantile(dist, clamp_quantile)
                low_val = -hi_val

                U = U.clamp(low_val, hi_val)
                Vh = Vh.clamp(low_val, hi_val)

                # move to store device and dtype (cpu and save_dtype by default)
                U = U.to(store_device, dtype=save_dtype).contiguous()
                Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

                lora_weights[key] = (U, Vh)
                del mat, U, S, Vh

                # release GPU memory if used
                if device_obj.type == "cuda":
                    torch.cuda.empty_cache()

    # make state dict for LoRA
    lora_sd = {}
    for key, (up_weight, down_weight) in lora_weights.items():
        lora_name = key.replace(".weight", "").replace(".", "_")
        lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + lora_name
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])  # same as rank

    # minimum metadata
    net_kwargs = {}
    metadata = {
        "ss_v2": str(False),
        "ss_base_model_version": flux_utils.MODEL_VERSION_FLUX_V1,
        "ss_network_module": "networks.lora_flux",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(lora_sd, False, False, False, True, False, time.time(), title, flux="dev")
        metadata.update(sai_metadata)

    save_to_file(save_to, lora_sd, metadata, save_dtype)

    logger.info(f"LoRA weights saved to {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はfloat",
    )
    parser.add_argument(
        "--model_org",
        type=str,
        default=None,
        required=True,
        help="Original model: safetensors file / 元モデル、safetensors",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        default=None,
        required=True,
        help="Tuned model, LoRA is difference of `original to tuned`: safetensors file / 派生モデル（生成されるLoRAは元→派生の差分になります）、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--mem_eff_safe_open",
        action="store_true",
        help="use memory efficient safe_open. This is an experimental feature, use only when memory is not enough."
        " / メモリ効率の良いsafe_openを使用する。実装は実験的なものなので、メモリが足りない場合のみ使用してください。",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file / 保存先のファイル名、safetensors",
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device to use for SVD computation: auto (prefer CUDA if available), cuda, or cpu",
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 値をクランプするための分位点、float、(0-1)。デフォルトは0.99",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    
    # New filtering options
    parser.add_argument(
        "--skip_double_blocks",
        action="store_true",
        help="Skip processing double block layers / double blockレイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_single_blocks",
        action="store_true",
        help="Skip processing single block layers / single blockレイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_kqv",
        action="store_true",
        help="Skip processing key, query, value attention layers (to_k, to_q, to_v) / キー、クエリ、バリューアテンションレイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_proj",
        action="store_true",
        help="Skip processing projection attention layers (to_out) / プロジェクションアテンションレイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_mlp0",
        action="store_true",
        help="Skip processing MLP layer 0 (mlp.0) / MLPレイヤー0の処理をスキップ",
    )
    parser.add_argument(
        "--skip_mlp2",
        action="store_true",
        help="Skip processing MLP layer 2 (mlp.2) / MLPレイヤー2の処理をスキップ",
    )
    parser.add_argument(
        "--skip_txt",
        action="store_true",
        help="Skip processing text-based keys / テキストベースのキーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_img",
        action="store_true",
        help="Skip processing image-based keys / 画像ベースのキーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_linear1",
        action="store_true",
        help="Skip processing linear1 layers / linear1レイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_linear2",
        action="store_true",
        help="Skip processing linear2 layers / linear2レイヤーの処理をスキップ",
    )
    parser.add_argument(
        "--skip_modulation",
        action="store_true",
        help="Skip processing modulation keys / モジュレーションキーの処理をスキップ",
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    svd(**vars(args))
