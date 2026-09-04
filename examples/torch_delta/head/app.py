"""Delta example (PyTorch, CPU), HEAD: the forward now normalizes its input.

The delta versus base/app.py is the `normalize` call, marked as a nested
region so the diff shows what it costs and in which torch kernels.

    drperf run -o out/head --state n=256,512 -- <venv>/bin/python examples/torch_delta/head/app.py
"""
import perfmark

st = perfmark.states(n=256, batch=64)
n, batch = int(st["n"]), int(st["batch"])

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
x = torch.randn(batch, n)
w = torch.randn(n, n)
bias = torch.randn(n)


def normalize(x):
    with perfmark.region("normalize", n=n):
        return F.normalize(x, dim=-1)


def forward(x):
    return normalize(x) @ w + bias


y = forward(x)  # warm-up outside any region
with perfmark.region("forward", n=n):
    y = forward(x)
print("head n=%d batch=%d y=%.4f" % (n, batch, float(y.sum())))
