import argparse
from pathlib import Path
import torch
import numpy as np


def cosine_distance(a, b, eps=1e-12):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a * b).sum(dim=-1)


def load_state(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


def maybe_get_ctx(state_dict):
    keys = [k for k in state_dict.keys() if k.endswith('prompt_learner.ctx') or k == 'prompt_learner.ctx' or k.endswith('.ctx')]
    if not keys:
        return None, None
    key = sorted(keys, key=len)[0]
    return key, state_dict[key].float().cpu()


def main():
    parser = argparse.ArgumentParser(description='Template script: 用 checkpoint 做 prompt drift / same-prompt MAP-vs-Bayes / full-cov 对比')
    parser.add_argument('--repo-root', required=False, help='BayesVLM repo root')
    parser.add_argument('--shot1-ckpt', required=False, help='1-shot checkpoint')
    parser.add_argument('--shot16-ckpt', required=False, help='16-shot checkpoint')
    parser.add_argument('--anchor-ckpt', required=False, help='Optional init/epoch0 checkpoint with original ctx')
    args = parser.parse_args()

    print('[Validation B] checkpoint 级别的 prompt drift quick check')
    if args.shot1_ckpt and args.shot16_ckpt:
        sd1 = load_state(args.shot1_ckpt)
        sd16 = load_state(args.shot16_ckpt)
        k1, ctx1 = maybe_get_ctx(sd1)
        k16, ctx16 = maybe_get_ctx(sd16)
        print(f'shot1 ctx key  = {k1}')
        print(f'shot16 ctx key = {k16}')
        if ctx1 is not None and ctx16 is not None and ctx1.shape == ctx16.shape:
            print(f'||ctx16-ctx1||_F = {(ctx16-ctx1).norm().item():.6f}')
            print(f'cosine_distance(flat(ctx16), flat(ctx1)) = {cosine_distance(ctx16.flatten(), ctx1.flatten()).item():.6f}')
        else:
            print('无法直接比较 ctx：checkpoint 中没找到 prompt_learner.ctx 或形状不一致。')
    else:
        print('传入 --shot1-ckpt 和 --shot16-ckpt 后可做 quick check。')

    print('\n[Validation A/E 模板]')
    print('同一个训练好的 checkpoint，分别评估：')
    print('1) model.forward_map_logits(batch)')
    print('2) model.forward_bayes_logits(batch)')
    print('3) 在 Bayes 下切换 use_full_cov=False / True')
    print('比较 ACC / NLPD / ECE。')
    print('由于这里依赖你的 repo 环境、dataset 构建和 args，我给出下面的伪代码模板：\n')

    template = r'''
# 伪代码：放到你的 repo 里运行
from bayesvlm.methods.text_only_bayes_coop import build_text_only_bayes_coop_model
from bayesvlm.methods.text_only_bayes_coop import evaluate_text_only_bayes_coop

# 1) 正常构建 model/state/ctx/test_loader
state = ...  # 从你的 recipe/build pipeline 得到
model = state["model"]
model.eval()

# 2) same-prompt: MAP vs Bayes
#    把 evaluate_text_only_bayes_coop 复制成两个版本：
#    - 一个用 prob_logits = model.forward_map_logits(batch)
#    - 一个用 prob_logits = model.forward_bayes_logits(batch)
#    再统一用 softmax / CE / ECE 计算

# 3) same-prompt: diag vs full-cov
model.use_full_cov = False
metrics_diag = eval_with_forward_bayes_logits(...)
model.use_full_cov = True
metrics_full = eval_with_forward_bayes_logits(...)

print(metrics_diag)
print(metrics_full)
'''
    print(template)


if __name__ == '__main__':
    main()
