#!/usr/bin/env python3
"""dit-score v0: fidelidade pós-quantização/pós-distilação para DiTs.

Compara um diretório de imagens VARIANT contra um diretório REFERENCE
(mesmos nomes de arquivo: {prompt_id}__{seed}.png), pareado por seed+prompt.

Métricas:
- LPIPS (AlexNet) vs referência        [análogo perceptual do KVDiv]
- PSNR vs referência
- ImageReward absoluto (variant e ref) [análogo do downstream eval]

Saída: JSON por par + agregados (média, desvio, piores pares).
Uso:
  python dit_score.py --reference DIR_REF --variant DIR_VAR \
      --prompts prompts.json --out result.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_pairs(ref_dir: Path, var_dir: Path):
    pairs = []
    for ref in sorted(ref_dir.glob("*.png")):
        var = var_dir / ref.name
        if var.exists():
            pairs.append((ref, var))
    return pairs


def to_tensor(img_path: Path, device):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device), img


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--variant", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips
    import ImageReward as RM

    prompts = {p["id"]: p["text"] for p in json.loads(args.prompts.read_text())["prompts"]}
    lpips_fn = lpips.LPIPS(net="alex").to(args.device)
    ir_model = RM.load("ImageReward-v1.0", device=args.device)

    pairs = load_pairs(args.reference, args.variant)
    if not pairs:
        raise SystemExit(f"nenhum par encontrado entre {args.reference} e {args.variant}")

    rows = []
    for ref_path, var_path in pairs:
        prompt_id = ref_path.stem.split("__")[0]
        prompt = prompts.get(prompt_id)
        ref_t, ref_img = to_tensor(ref_path, args.device)
        var_t, var_img = to_tensor(var_path, args.device)
        with torch.no_grad():
            lp = lpips_fn(ref_t * 2 - 1, var_t * 2 - 1).item()
            row = {
                "file": ref_path.name,
                "prompt_id": prompt_id,
                "lpips": round(lp, 4),
                "psnr": round(psnr(ref_t, var_t), 2),
            }
            if prompt:
                row["ir_ref"] = round(ir_model.score(prompt, [str(ref_path)]), 4)
                row["ir_var"] = round(ir_model.score(prompt, [str(var_path)]), 4)
                row["ir_delta"] = round(row["ir_var"] - row["ir_ref"], 4)
        rows.append(row)
        print(row)

    lp_all = [r["lpips"] for r in rows]
    ps_all = [r["psnr"] for r in rows if r["psnr"] != float("inf")]
    ird_all = [r["ir_delta"] for r in rows if "ir_delta" in r]
    summary = {
        "n_pairs": len(rows),
        "lpips_mean": round(float(np.mean(lp_all)), 4),
        "lpips_std": round(float(np.std(lp_all)), 4),
        "psnr_mean": round(float(np.mean(ps_all)), 2),
        "ir_delta_mean": round(float(np.mean(ird_all)), 4) if ird_all else None,
        "worst_lpips": sorted(rows, key=lambda r: -r["lpips"])[:3],
    }
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\nSUMMARY:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
