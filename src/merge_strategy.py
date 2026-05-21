import os
import time
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def _base_key(n: str) -> str:
    return n.replace("lora_A.", "").replace("lora_B.", "")


def _return_deltas_by_key(tv):
    deltas_by_key: dict[str, list[torch.Tensor]] = {}
    for _, params in tv.items():
        # Collect keys
        # keys_delta: those not related to LoRA but full weights (if any)
        keys_delta = {k for k in params if not ("lora_A." in k or "lora_B." in k)}
        keys_A = {k for k in params if "lora_A." in k}
        keys_B = {k for k in params if "lora_B." in k}
        common = {_base_key(k) for k in keys_A} & {_base_key(k) for k in keys_B}
        for bk in common:
            a_key = next(k for k in keys_A if _base_key(k) == bk)
            b_key = next(k for k in keys_B if _base_key(k) == bk)
            original_device = params[b_key].device
            params_B = params[b_key].to("cuda")
            params_A = params[a_key].to("cuda")
            delta = (params_B @ params_A).to(original_device)
            deltas_by_key.setdefault(bk, []).append(delta)
        for dk in keys_delta:
            delta = params[dk]
            deltas_by_key.setdefault(dk, []).append(delta)

    return deltas_by_key


# ------------------------------
# Linear
# ------------------------------


def Linear(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'average'
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    A_acc: dict[str, torch.Tensor] = {}
    B_acc: dict[str, torch.Tensor] = {}
    for _, params in tv.items():
        for name, t in params.items():
            if "lora_A." in name:
                k = _base_key(name)
                A_acc[k] = t.clone() if k not in A_acc else A_acc[k].add_(t)
            elif "lora_B." in name:
                k = _base_key(name)
                B_acc[k] = t.clone() if k not in B_acc else B_acc[k].add_(t)

    merged: dict[str, torch.Tensor] = {}
    for k in A_acc.keys() & B_acc.keys():
        merged[k] = B_acc[k] @ A_acc[k]
    if merging_type == "mean":
        n = len(tv)
        if n > 0:
            for k in merged:
                merged[k].div_(n)

    end_time = time.time()
    # print some info about the merging process
    print(
        f"Linear merging with merging_type={merging_type}, "
        f"scaling_factor={scaling_factor} took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    # apply scaling factor
    for k in A_acc.keys() & B_acc.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# Task Arithmetic
# ------------------------------


def TA(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'mean'
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    merged: dict[str, torch.Tensor] = {}
    for _, params in tv.items():
        keys_A = {k for k in params if "lora_A." in k}
        keys_B = {k for k in params if "lora_B." in k}
        common = {_base_key(k) for k in keys_A} & {_base_key(k) for k in keys_B}
        for bk in common:
            a_key = next(k for k in keys_A if _base_key(k) == bk)
            b_key = next(k for k in keys_B if _base_key(k) == bk)
            delta = params[b_key] @ params[a_key]
            if bk not in merged:
                merged[bk] = delta.clone()
            else:
                merged[bk].add_(delta)
    if merging_type == "mean":
        n = len(tv)
        if n > 0:
            for k in merged:
                merged[k].div_(n)
    end_time = time.time()
    # print some info about the merging process
    print(
        f"TA merging with merging_type={merging_type}, "
        f"scaling_factor={scaling_factor} took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    # apply scaling factor
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# SVD (shared U, s; plain mean/sum over sV)
# ------------------------------


def SVD(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'mean' | 'sum'
    svd_tol: float = 1e-5,
    scaling_factor: float = 1.0,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in deltas_by_key.items():
        R, C = lst[0].shape
        M = torch.cat(lst, dim=1)
        U, s, Vh = torch.linalg.svd(M.to(torch.float64), full_matrices=False)
        U, s, Vh = U.to(torch.float32), s.to(torch.float32), Vh.to(torch.float32)

        keep = s > svd_tol
        if keep.sum() == 0:
            merged[bk] = torch.zeros_like(lst[0])
            continue
        U = U[:, keep]
        s = s[keep]
        Vh = Vh[keep, :]

        s_diag = torch.diag(s)
        sV_tasks = []
        for t in range(T):
            Vh_t = Vh[:, t * C : (t + 1) * C]
            sV_t = s_diag @ Vh_t
            sV_tasks.append(sV_t)
        S = torch.stack(sV_tasks, dim=0)

        if merging_type == "sum":
            sV_merged = S.sum(dim=0)
        else:
            sV_merged = S.mean(dim=0)

        delta_merged = U @ sV_merged
        merged[bk] = delta_merged
    end_time = time.time()
    # print some info about the merging process
    print(
        f"SVD merging with svd_tol={svd_tol}, "
        f"merging_type={merging_type}, scaling_factor={scaling_factor} "
        f"took {end_time - start_time:.2f} seconds."
    )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor

    return merged


# ------------------------------
# Ties + DareTies
# ------------------------------


def Ties(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",
    topK: float = 1.0,
    drop_p: float = 0.0,
    seed: Optional[int] = None,
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged

    def _rowwise_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        if k >= x.shape[1]:
            return torch.ones_like(x, dtype=torch.bool)
        _, idx = x.abs().topk(k, dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _parse_topk(D: int, tk: float) -> int:
        if tk > 1:
            tk = tk / 100.0
        tk = float(max(0.0, min(1.0, tk)))
        return D if tk >= 1.0 else max(1, int(round(D * tk)))

    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_p = 1.0 - drop_p
    scale = 1.0 / keep_p if drop_p > 0 else 1.0

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in tqdm(deltas_by_key.items()):
        stack = torch.stack(lst, dim=0)
        flat = stack.reshape(T, -1)

        # Top-K
        k = _parse_topk(flat.shape[1], topK)
        topk_mask = _rowwise_topk_mask(flat, k)
        flat = flat * topk_mask.float()

        # DARE
        if drop_p > 0:
            bern = torch.bernoulli(torch.full_like(flat, keep_p), generator=gen)
            flat = flat * bern * scale

        sign_sum = flat.sign().sum(dim=0)
        sign_sum[sign_sum == 0] = 1
        rows_keep = torch.where(sign_sum.unsqueeze(0) > 0, flat > 0, flat < 0)
        selected = flat * rows_keep.float()

        if merging_type == "mean":
            nonzero = (selected != 0).sum(dim=0).clamp_min(1).float()
            agg = selected.sum(dim=0) / nonzero
        else:  # 'sum'
            agg = selected.sum(dim=0)

        merged[bk] = agg.reshape(stack.shape[1:])
    end_time = time.time()
    # print some info about the merging process
    if drop_p > 0:
        print(
            f"Dare-Ties merging with topK={topK}, drop_p={drop_p}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    else:
        print(
            f"Ties merging with topK={topK}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)

    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# KnOTS (Ties+DareTies)
# ------------------------------


def KnOTS(
    tv: dict[str, dict[str, torch.Tensor]],
    merging_type: str = "sum",  # 'sum' | 'mean'
    drop_p: float = 0.0,
    seed: Optional[int] = None,
    svd_tol: float = 1e-5,
    topK: float = 1.0,
    scaling_factor: float = 0.3,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        merged = load_file(cache_path)
        for k in merged.keys():
            merged[k] = merged[k] * scaling_factor
        return merged
    start_time = time.time()
    deltas_by_key = _return_deltas_by_key(tv)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_p = 1.0 - drop_p
    scale = 1.0 / keep_p if keep_p > 0 else 0.0

    def _rowwise_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        if k >= x.shape[1]:
            return torch.ones_like(x, dtype=torch.bool)
        _, idx = x.abs().topk(k, dim=1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def _parse_topk(TD: int, tk: float) -> int:
        if tk > 1:
            tk = tk / 100.0
        tk = float(max(0.0, min(1.0, tk)))
        return max(1, int(round(TD * tk))) if tk < 1.0 else TD

    merged: dict[str, torch.Tensor] = {}
    T = len(tv)

    for bk, lst in tqdm(deltas_by_key.items()):
        original_device = lst[0].device
        lst = [t.to("cuda") for t in lst]
        R, C = lst[0].shape
        M = torch.cat(lst, dim=1)
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        U, s, Vh = U.float(), s.float(), Vh.float()

        keep = s > svd_tol
        if keep.sum() == 0:
            merged[bk] = torch.zeros_like(lst[0])
            continue
        U, s, Vh = U[:, keep], s[keep], Vh[keep, :]

        s_diag = torch.diag(s)
        sV_tasks: list[torch.Tensor] = []
        for t in range(T):
            Vh_t = Vh[:, t * C : (t + 1) * C]
            sV_t = s_diag @ Vh_t
            sV_tasks.append(sV_t)
        S = torch.stack(sV_tasks, dim=0)

        # --- Top-K ---
        flat = S.reshape(T, -1)
        k = _parse_topk(flat.shape[1], topK)
        topk_mask = _rowwise_topk_mask(flat, k)
        flat = flat * topk_mask.float()

        # --- DARE  ---
        if drop_p > 0:
            bern = torch.bernoulli(torch.full_like(flat, keep_p), generator=gen)
            flat = flat * bern * scale

        sign_sum = flat.sign().sum(dim=0)
        sign_sum[sign_sum == 0] = 1
        rows_keep = torch.where(sign_sum.unsqueeze(0) > 0, flat > 0, flat < 0)
        selected = flat * rows_keep.float()

        if merging_type == "mean":
            nonzero = (selected != 0).sum(dim=0).clamp_min(1).float()
            agg = selected.sum(dim=0) / nonzero
        else:  # 'sum'
            agg = selected.sum(dim=0)

        sV_merged = agg.reshape(S.shape[1], S.shape[2])
        merged[bk] = (U @ sV_merged).to(original_device)
    end_time = time.time()

    # print some info about the merging process
    if drop_p > 0:
        print(
            f"KnOTS-Dare-Ties merging with topK={topK}, drop_p={drop_p}, svd_tol={svd_tol}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    else:
        print(
            f"KnOTS-Ties merging with topK={topK}, svd_tol={svd_tol}, "
            f"merging_type={merging_type}, seed={seed}, scaling_factor={scaling_factor} "
            f"took {end_time - start_time:.2f} seconds."
        )
    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)
    for k in merged.keys():
        merged[k] = merged[k] * scaling_factor
    return merged


# ------------------------------
# LoRA-LEGO
# ------------------------------
def LoRA_LEGO(
    tv: dict[str, dict[str, torch.Tensor]],
    K: Optional[int] = None,
    kmeans_iters: int = 10,
    use_param_reweighting: bool = True,
    use_output_reweighting: bool = True,
    seed: Optional[int] = None,
    cache_path: Optional[str] = None,
    profile: bool = False,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and os.path.exists(cache_path) and not profile:
        print(f"Loading cached merged TV from {cache_path}...")
        # Load safetensor
        return load_file(cache_path)
    _EPS = 1e-8

    def _base_AB(params: dict[str, torch.Tensor], bk: str):
        a_key = next(k for k in params if "lora_A." in k and _base_key(k) == bk)
        b_key = next(k for k in params if "lora_B." in k and _base_key(k) == bk)
        return params[a_key], params[b_key]

    def _normalize(v):
        return v / (v.norm() + _EPS)

    def _kmeans(x: torch.Tensor, K: int, iters: int, gen: Optional[torch.Generator]):
        N, D = x.shape
        if K >= N:
            return torch.arange(N, device=x.device), x.clone()
        idx = torch.randperm(N, generator=gen, device=x.device)[:K]
        cent = x[idx].clone()
        for _ in range(iters):
            d = (x.unsqueeze(1) - cent.unsqueeze(0)).pow(2).sum(2)
            assign = d.argmin(1)
            for k in range(K):
                m = assign == k
                if m.any():
                    cent[k] = x[m].mean(0)
        return assign, cent

    start_time = time.time()
    gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed else None
    merged = {}

    task_keys = list(tv.keys())
    bases_A = [{_base_key(k) for k in tv[t] if "lora_A." in k} for t in task_keys]
    bases_B = [{_base_key(k) for k in tv[t] if "lora_B." in k} for t in task_keys]
    common_bases = set.intersection(*bases_A) & set.intersection(*bases_B)

    for bk in tqdm(common_bases):
        AB = [_base_AB(tv[t], bk) for t in task_keys]
        r, d_in = AB[0][0].shape
        d_out, rB = AB[0][1].shape
        assert r == rB
        KK = K or r

        # build raw pool
        feat_list, raw_list = [], []
        for A, B in AB:
            for k in range(r):
                a_vec = A[k]
                b_vec = B[:, k]
                feat = torch.cat([_normalize(a_vec), _normalize(b_vec)])
                raw = torch.cat([a_vec, b_vec])
                feat_list.append(feat)
                raw_list.append(raw)

        features = torch.stack(feat_list)
        raw_pool = torch.stack(raw_list)
        labels, _ = _kmeans(features, KK, kmeans_iters, gen)

        merged_delta = torch.zeros((d_out, d_in), dtype=raw_pool.dtype, device=raw_pool.device)
        for cid in range(KK):
            m = labels == cid
            if not m.any():
                continue
            mu = raw_pool[m].mean(0)
            if use_param_reweighting:
                eps = 1e-12
                inf_norm = mu.abs().amax()
                avg_inf = raw_pool[m].abs().amax(dim=1).mean()
                mu = mu * (avg_inf / (inf_norm + eps))
            a_bar = mu[:d_in]
            b_bar = mu[d_in:]
            merged_delta.add_(torch.outer(b_bar, a_bar))

        if use_output_reweighting and KK > 0:
            merged_delta = merged_delta * ((r**0.5) / (KK**0.5))

        merged[bk] = merged_delta
    end_time = time.time()

    # print some info about the merging process
    print(
        f"LoRA-LEGO merging with K={K}, kmeans_iters={kmeans_iters}, "
        f"use_param_reweighting={use_param_reweighting}, use_output_reweighting={use_output_reweighting}, seed={seed}"  # noqa
        f" took {end_time - start_time:.2f} seconds."
    )

    if cache_path is not None and not profile:
        print(f"Caching merged TV to {cache_path}...")
        # Save safetensor
        save_file(merged, cache_path)

    return merged
