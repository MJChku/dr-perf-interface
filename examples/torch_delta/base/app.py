"""Delta example (PyTorch, CPU), BASE: a linear layer forward.

    drperf run -o out/base --state n=256,512 -- <venv>/bin/python examples/torch_delta/base/app.py
"""
import perfmark

st = perfmark.states(n=256, batch=64)
n, batch = int(st["n"]), int(st["batch"])

import torch  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
x = torch.randn(batch, n)
w = torch.randn(n, n)
bias = torch.randn(n)


def forward(x):
    return x @ w + bias


y = forward(x)  # warm-up outside any region
with perfmark.region("forward", n=n):
    y = forward(x)
print("base n=%d batch=%d y=%.4f" % (n, batch, float(y.sum())))
