import argparse
import math
import torch
import numpy as np


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            rows.append((lo, hi, 0, np.nan, np.nan, np.nan))
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        frac = mask.mean()
        ece += abs(bin_acc - bin_conf) * frac
        rows.append((lo, hi, int(mask.sum()), float(bin_acc), float(bin_conf), float(abs(bin_acc - bin_conf))))
    return float(ece), rows


def entropy(probs: np.ndarray):
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def rankdata(x: np.ndarray):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for uidx, cnt in enumerate(counts):
        if cnt > 1:
            idx = np.where(inv == uidx)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray):
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float('nan')
    return float((rx * ry).sum() / denom)


def auc_roc(scores: np.ndarray, labels01: np.ndarray):
    pos = scores[labels01 == 1]
    neg = scores[labels01 == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    scores_all = np.concatenate([pos, neg])
    labels_all = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    ranks = rankdata(scores_all) + 1.0
    pos_ranks = ranks[labels_all == 1]
    u = pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def probit_probs(logits_mean: np.ndarray, logits_var: np.ndarray):
    adj = logits_mean / np.sqrt(1.0 + (math.pi / 8.0) * np.clip(logits_var, 0.0, None))
    adj = adj - adj.max(axis=1, keepdims=True)
    ex = np.exp(adj)
    return ex / ex.sum(axis=1, keepdims=True)


def mc_probs(logits_mean: np.ndarray, logits_var: np.ndarray, num_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    mean = logits_mean[:, None, :]
    std = np.sqrt(np.clip(logits_var[:, None, :], 0.0, None))
    eps = rng.standard_normal(size=(logits_mean.shape[0], num_samples, logits_mean.shape[1]))
    samples = mean + std * eps
    samples = samples - samples.max(axis=2, keepdims=True)
    ex = np.exp(samples)
    probs = ex / ex.sum(axis=2, keepdims=True)
    return probs.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to test_predictions.pt')
    parser.add_argument('--mc-samples', type=int, default=1000)
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    obj = torch.load(args.pred, map_location='cpu')
    required = ['probs', 'labels', 'preds', 'logits_mean', 'logits_var']
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f'Missing keys in {args.pred}: {missing}. Available: {list(obj.keys())}')

    probs = to_numpy(obj['probs']).astype(np.float64)
    labels = to_numpy(obj['labels']).astype(np.int64)
    preds = to_numpy(obj['preds']).astype(np.int64)
    logits_mean = to_numpy(obj['logits_mean']).astype(np.float64)
    logits_var = to_numpy(obj['logits_var']).astype(np.float64)

    if probs.ndim != 2 or logits_mean.ndim != 2 or logits_var.ndim != 2:
        raise ValueError(
            f'Expect probs/logits_mean/logits_var to be rank-2 [N,C], '
            f'got probs{probs.shape}, logits_mean{logits_mean.shape}, logits_var{logits_var.shape}'
        )
    if not (len(labels) == len(preds) == probs.shape[0] == logits_mean.shape[0] == logits_var.shape[0]):
        raise ValueError('Inconsistent first dimension among labels/preds/probs/logits_mean/logits_var')

    correct = (preds == labels).astype(np.int64)
    row_idx = np.arange(len(preds))
    pred_logit_var = logits_var[row_idx, preds]
    pred_logit_mean = logits_mean[row_idx, preds]
    true_logit_var = logits_var[row_idx, labels]
    true_logit_mean = logits_mean[row_idx, labels]

    print(f'N={len(labels)}, C={probs.shape[1]}')
    print('Loaded keys:', list(obj.keys()))

    print('\n[Validation C] 方差是否真的识别错误样本')
    var_correct = pred_logit_var[correct == 1]
    var_wrong = pred_logit_var[correct == 0]
    print(f'mean(pred_logit_var | correct=1) = {var_correct.mean():.6f}')
    print(f'mean(pred_logit_var | correct=0) = {var_wrong.mean():.6f}')
    print(f'median(pred_logit_var | correct=1) = {np.median(var_correct):.6f}')
    print(f'median(pred_logit_var | correct=0) = {np.median(var_wrong):.6f}')
    err = 1 - correct
    print(f'Spearman(pred_logit_var, error) = {spearman_corr(pred_logit_var, err):.6f}')
    print(f'AUROC(score=pred_logit_var, target=error) = {auc_roc(pred_logit_var, err):.6f}')

    print('\n[辅助] 预测类 vs 真实类 的 logit 统计')
    print(f'mean(pred_logit_mean) = {pred_logit_mean.mean():.6f}')
    print(f'mean(true_logit_mean) = {true_logit_mean.mean():.6f}')
    print(f'mean(pred_logit_var)  = {pred_logit_var.mean():.6f}')
    print(f'mean(true_logit_var)  = {true_logit_var.mean():.6f}')

    print('\n[辅助] 正确/错误样本的 confidence 对比')
    conf = probs.max(axis=1)
    print(f'mean(confidence | correct=1) = {conf[correct==1].mean():.6f}')
    print(f'mean(confidence | correct=0) = {conf[correct==0].mean():.6f}')
    print(f'mean(entropy | correct=1) = {entropy(probs)[correct==1].mean():.6f}')
    print(f'mean(entropy | correct=0) = {entropy(probs)[correct==0].mean():.6f}')

    print('\n[Validation D] 解析 probit vs MC 近似')
    probs_probit = probit_probs(logits_mean, logits_var)
    ece_probit, _ = ece_score(probs_probit, labels, n_bins=args.bins)
    probs_mc = mc_probs(logits_mean, logits_var, num_samples=args.mc_samples, seed=args.seed)
    ece_mc, _ = ece_score(probs_mc, labels, n_bins=args.bins)
    print(f'ECE(saved probs, bins={args.bins})   = {ece_score(probs, labels, n_bins=args.bins)[0]:.6f}')
    print(f'ECE(probit, bins={args.bins})       = {ece_probit:.6f}')
    print(f'ECE(MC-{args.mc_samples}, bins={args.bins}) = {ece_mc:.6f}')

    print('\n[辅助] 逐置信区间 reliability 明细 (saved probs)')
    _, rows = ece_score(probs, labels, n_bins=args.bins)
    for lo, hi, count, acc, avg_conf, gap in rows:
        print(f'bin=[{lo:.2f},{hi:.2f}) n={count:5d} acc={acc!s:>8} conf={avg_conf!s:>8} gap={gap!s:>8}')

    print('\n[说明]')
    print('- 你的 .pt tensor_payload 里只有 labels/preds/probs/logits_mean/logits_var；')
    print('  correct 和 pred_logit_var 只在 jsonl 行级输出里有，因此这里是从 preds/labels/logits_var 现算的。')
    print('- pred_logit_var = logits_var[n, preds[n]]，表示“模型最终预测类别”的 logit 方差。')
    print('- 如果 mean(pred_logit_var | wrong) 没明显高于 correct，说明 text-only uncertainty 没抓住错误样本。')
    print('- 如果 ECE(MC) 明显优于 ECE(probit)，说明解析 probit 近似在高 shot 下可能有失真。')


if __name__ == '__main__':
    main()
