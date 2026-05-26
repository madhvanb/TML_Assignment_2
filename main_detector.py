import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm


REPO_ID = "SprintML/tml26_task2"
N_SUSPECTS = 360
NUM_PROBE = 5000
HARD_SUBSET_SIZE = 1024
BATCH_SIZE = 128
NOISE_LEVELS = (0.02, 0.05, 0.10)

ROOT = Path("/home/atml_team011/tml26-task2")
CACHE_DIR = ROOT / "model_cache"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "all_metrics.csv"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

ZERO_ROW_KEYS = [
    "w_cos", "exact_copy", "w_layer_mean",
    "o_top1", "o_top5", "o_klsim", "o_spearman", "o_top1_gap",
    "di_loss_in", "di_loss_out", "di_gap",
    "cka_mean",
    "hard_top1", "hard_top5", "hard_prob_cos", "hard_conf_cos",
    "hard_robustness_gap",
    "multi_signal_score", "hard_score",
]


def make_model():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 100)
    return m


def load_weights(path):
    sd = load_file(path, device="cpu")
    model = make_model()
    model.load_state_dict(sd, strict=False)
    return model, sd


def download(filename):
    return hf_hub_download(repo_id=REPO_ID, filename=filename, cache_dir=str(CACHE_DIR))


def standard_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_in_loader(idx_path):
    train_full = datasets.CIFAR100(
        root=str(DATA_DIR), train=True, download=True, transform=standard_transform()
    )
    with open(idx_path) as f:
        indices = list(json.load(f))[:NUM_PROBE]
    return DataLoader(
        Subset(train_full, indices),
        batch_size=200, shuffle=False, num_workers=4, pin_memory=True,
    )


def get_out_loader(idx_path):
    train_full = datasets.CIFAR100(
        root=str(DATA_DIR), train=True, download=True, transform=standard_transform()
    )
    with open(idx_path) as f:
        in_set = set(json.load(f))
    out_indices = [i for i in range(len(train_full)) if i not in in_set][:NUM_PROBE]
    return DataLoader(
        Subset(train_full, out_indices),
        batch_size=200, shuffle=False, num_workers=4, pin_memory=True,
    )


def select_hard_subset(target_model):
    test = datasets.CIFAR100(
        root=str(DATA_DIR), train=False, download=True, transform=standard_transform()
    )
    loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    target_model = target_model.to(DEVICE)
    ents = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="entropy"):
            x = x.to(DEVICE)
            p = F.softmax(target_model(x), dim=1)
            e = -(p * p.clamp(min=1e-8).log()).sum(dim=1)
            ents.append(e.cpu())
    ents = torch.cat(ents).numpy()
    hard_idx = np.argsort(ents)[-HARD_SUBSET_SIZE:].tolist()
    return Subset(test, hard_idx)


def hard_loader(subset):
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def get_logits(model, loader):
    model = model.to(DEVICE).eval()
    parts = []
    with torch.no_grad():
        for x, _ in loader:
            parts.append(model(x.to(DEVICE)).cpu())
    return torch.cat(parts)


def get_labels(loader):
    parts = []
    for _, y in loader:
        parts.append(y)
    return torch.cat(parts).long()


def collect_probs(model, loader):
    model = model.to(DEVICE).eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            p = F.softmax(model(x), dim=1)
            probs.append(p.cpu())
    return torch.cat(probs)


def noise_stability(model, loader, levels=NOISE_LEVELS):
    model = model.to(DEVICE).eval()
    stabs = []
    with torch.no_grad():
        for sigma in levels:
            same = 0
            total = 0
            for x, _ in loader:
                x = x.to(DEVICE)
                clean = model(x).argmax(1)
                noisy = model(x + torch.randn_like(x) * sigma).argmax(1)
                same += (clean == noisy).sum().item()
                total += clean.numel()
            stabs.append(same / total)
    return float(np.mean(stabs))


def weight_metrics(t_sd, s_sd):
    common = [k for k in t_sd if k in s_sd and t_sd[k].shape == s_sd[k].shape]
    t_vec = torch.cat([t_sd[k].float().flatten() for k in common])
    s_vec = torch.cat([s_sd[k].float().flatten() for k in common])
    cos = F.cosine_similarity(t_vec.unsqueeze(0), s_vec.unsqueeze(0)).item()
    exact = float((t_vec - s_vec).abs().max().item() < 1e-4)
    layer_cos = []
    for k in common:
        t = t_sd[k].float().flatten()
        s = s_sd[k].float().flatten()
        layer_cos.append(F.cosine_similarity(t.unsqueeze(0), s.unsqueeze(0)).item())
    return {
        "w_cos": cos,
        "exact_copy": exact,
        "w_layer_mean": float(np.mean(layer_cos)),
    }


def output_metrics(t_logits, s_logits):
    top1 = (t_logits.argmax(1) == s_logits.argmax(1)).float().mean().item()
    t5 = t_logits.topk(5, dim=1).indices
    s5 = s_logits.topk(5, dim=1).indices
    top5 = float(np.mean([
        len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(t5, s5)
    ]))
    tp = F.softmax(t_logits.float(), dim=1).clamp(1e-10)
    sp = F.softmax(s_logits.float(), dim=1).clamp(1e-10)
    kl = F.kl_div(sp.log(), tp, reduction="batchmean").item()
    t_ranks = t_logits.float().argsort(1).argsort(1).float()
    s_ranks = s_logits.float().argsort(1).argsort(1).float()
    t_c = t_ranks - t_ranks.mean(1, keepdim=True)
    s_c = s_ranks - s_ranks.mean(1, keepdim=True)
    num = (t_c * s_c).sum(1)
    denom = t_c.norm(dim=1) * s_c.norm(dim=1) + 1e-10
    spearman = (num / denom).mean().item()
    return {
        "o_top1": top1,
        "o_top5": top5,
        "o_klsim": 1.0 / (1.0 + kl),
        "o_spearman": spearman,
    }


CKA_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


class Hook:
    def __init__(self, module):
        self.out = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _input, output):
        self.out = output.detach().cpu()

    def remove(self):
        self.handle.remove()


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = (X @ Y.T).norm(p="fro").item() ** 2
    denom = (X @ X.T).norm(p="fro").item() * (Y @ Y.T).norm(p="fro").item()
    return num / (denom + 1e-10)


def collect_features(model, loader):
    model = model.to(DEVICE).eval()
    children = dict(model.named_children())
    hooks = {l: Hook(children[l]) for l in CKA_LAYERS}
    feats = {l: [] for l in CKA_LAYERS}
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
            for l in CKA_LAYERS:
                f = hooks[l].out
                if f.dim() == 4:
                    f = f.mean((-2, -1))
                feats[l].append(f)
    for h in hooks.values():
        h.remove()
    return {l: torch.cat(feats[l]).float() for l in CKA_LAYERS}


def cka_metrics(target_feats, suspect_feats):
    scores = [linear_cka(target_feats[l], suspect_feats[l]) for l in CKA_LAYERS]
    return {"cka_mean": float(np.mean(scores))}


def hard_probe_metrics(t_probs, s_probs, t_stab, s_stab):
    t_top1 = t_probs.argmax(1)
    s_top1 = s_probs.argmax(1)
    top1 = (t_top1 == s_top1).float().mean().item()

    t5 = t_probs.topk(5, dim=1).indices
    s5 = s_probs.topk(5, dim=1).indices
    top5 = float(np.mean([
        len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(t5, s5)
    ]))

    prob_cos = F.cosine_similarity(t_probs, s_probs, dim=1).mean().item()

    t_conf = t_probs.max(1).values
    s_conf = s_probs.max(1).values
    conf_cos = F.cosine_similarity(t_conf.unsqueeze(0), s_conf.unsqueeze(0)).item()

    rob_gap = 1.0 - abs(t_stab - s_stab)

    return {
        "hard_top1": top1,
        "hard_top5": top5,
        "hard_prob_cos": prob_cos,
        "hard_conf_cos": conf_cos,
        "hard_robustness_gap": rob_gap,
    }


def compute_scores(rows):
    df = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}

    def norm(k):
        x = df[k]
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    multi_signal = (
        0.28 * norm("cka_mean")
        + 0.24 * norm("di_gap")
        + 0.14 * norm("o_top1_gap")
        + 0.10 * norm("o_top1")
        + 0.08 * norm("w_cos")
        + 0.06 * norm("o_klsim")
        + 0.04 * norm("o_spearman")
        + 0.03 * norm("o_top5")
        + 0.03 * norm("w_layer_mean")
    )
    multi_signal += df["exact_copy"] * 3.0
    multi_signal = (multi_signal - multi_signal.min()) / (multi_signal.max() - multi_signal.min() + 1e-10)

    hard = (
        0.40 * norm("hard_top1")
        + 0.25 * norm("hard_top5")
        + 0.35 * norm("hard_prob_cos")
    )

    final = 0.70 * multi_signal + 0.30 * hard
    final = (final - final.min()) / (final.max() - final.min() + 1e-10)

    return final, multi_signal, hard


def save_metrics(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"metrics saved -> {path}")


def save_submission(ids, scores, path):
    scores = np.asarray(scores, dtype=np.float64)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for sid, sc in zip(ids, scores):
            w.writerow([int(sid), float(sc)])
    print(f"saved {len(ids)} rows -> {path}")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE}  multi-probe: {NUM_PROBE}  hard-probe: {HARD_SUBSET_SIZE}  suspects: {N_SUSPECTS}")

    print("loading target")
    t_path = download("target_model/weights.safetensors")
    idx_path = download("target_model/train_main_idx.json")
    t_model, t_sd = load_weights(t_path)

    print("building multi-signal probe loaders")
    in_loader = get_in_loader(idx_path)
    out_loader = get_out_loader(idx_path)

    print("selecting hard probe subset")
    hard_subset = select_hard_subset(t_model)
    h_loader = hard_loader(hard_subset)

    print("computing target signals")
    t_logits_in = get_logits(t_model, in_loader)
    t_logits_out = get_logits(t_model, out_loader)
    t_feats = collect_features(t_model, in_loader)
    t_preds_out = t_logits_out.argmax(1)
    y_in = get_labels(in_loader)
    y_out = get_labels(out_loader)
    t_probs_hard = collect_probs(t_model, h_loader)
    t_stab_hard = noise_stability(t_model, h_loader)
    print(f"target hard-probe stability: {t_stab_hard:.4f}")
    del t_model

    print(f"scoring {N_SUSPECTS} suspects")
    rows = []
    for i in tqdm(range(N_SUSPECTS)):
        try:
            s_path = download(f"suspect_models/suspect_{i:03d}.safetensors")
            s_model, s_sd = load_weights(s_path)
        except Exception as e:
            print(f"failed to load suspect {i}: {e}")
            rows.append({"id": i, **{k: 0.0 for k in ZERO_ROW_KEYS}})
            continue

        row = {"id": i}
        row.update(weight_metrics(t_sd, s_sd))

        s_logits_in = get_logits(s_model, in_loader)
        row.update(output_metrics(t_logits_in, s_logits_in))

        s_logits_out = get_logits(s_model, out_loader)
        top1_out = (t_preds_out == s_logits_out.argmax(1)).float().mean().item()
        row["o_top1_gap"] = row["o_top1"] - top1_out

        ce_in = F.cross_entropy(s_logits_in.float(), y_in, reduction="mean").item()
        ce_out = F.cross_entropy(s_logits_out.float(), y_out, reduction="mean").item()
        row["di_loss_in"] = ce_in
        row["di_loss_out"] = ce_out
        row["di_gap"] = ce_out - ce_in

        s_feats = collect_features(s_model, in_loader)
        row.update(cka_metrics(t_feats, s_feats))

        s_probs_hard = collect_probs(s_model, h_loader)
        s_stab_hard = noise_stability(s_model, h_loader)
        row.update(hard_probe_metrics(t_probs_hard, s_probs_hard, t_stab_hard, s_stab_hard))

        row["multi_signal_score"] = 0.0
        row["hard_score"] = 0.0

        del s_model, s_sd, s_logits_in, s_logits_out, s_feats, s_probs_hard
        rows.append(row)

    final, multi, hard = compute_scores(rows)
    for r, m, h in zip(rows, multi, hard):
        r["multi_signal_score"] = float(m)
        r["hard_score"] = float(h)

    save_metrics(rows, METRICS_PATH)

    ids = [r["id"] for r in rows]
    save_submission(ids, final, OUTPUT_PATH)

    ranked = sorted(zip(ids, final), key=lambda x: -x[1])
    print("top 20:")
    for sid, sc in ranked[:20]:
        r = rows[sid]
        print(f"  {sid:3d}  score={sc:.4f}  cos={r['w_cos']:.4f}  "
              f"cka={r['cka_mean']:.4f}  di_gap={r['di_gap']:.4f}  "
              f"hard_top1={r['hard_top1']:.4f}  exact={int(r['exact_copy'])}")

    print(f"exact copies: {sum(r['exact_copy'] for r in rows):.0f}")
    print(f"final score mean/std: {final.mean():.4f} / {final.std():.4f}")


if __name__ == "__main__":
    main()
