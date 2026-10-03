"""Check the saved planar-output patch against the original tensor2video method."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from einops import rearrange

p = argparse.ArgumentParser()
p.add_argument('--tree', type=Path, required=True)
p.add_argument('--measure-pil', action='store_true', help='Measure the actual post-astype PIL tail with drperf')
a = p.parse_args()
relative = 'telefuser/core/base_pipeline.py'
original = subprocess.check_output(['git', '-C', str(a.tree), 'show', 'HEAD:' + relative], text=True)
patch = Path(__file__).resolve().parent / 'cpu_patches/telefuser_planar_output.patch'
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text(original)
    subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
    candidate = target.read_text()


def extract(source):
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'BasePipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2video')
    namespace = dict(torch=torch, np=np, Image=Image, F=F, rearrange=rearrange,
                     logger=SimpleNamespace(info=lambda *args: None))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<tensor2video>', 'exec'), namespace)
    return namespace['tensor2video']


before, after = extract(original), extract(candidate)
torch.set_num_threads(1)
torch.manual_seed(42)
if a.measure_pil:
    import perfmark

    def pil_tail(source):
        module = ast.parse(source)
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'BasePipeline')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'tensor2video')
        boundary = next(i for i, n in enumerate(method.body)
                        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'astype')
        function = ast.parse('def convert(frames):\n    pass\n').body[0]
        function.body = method.body[boundary + 1:]
        namespace = dict(Image=Image)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     '<actual PIL conversion tail>', 'exec'), namespace)
        return namespace['convert']

    variants = [('abot_pil_before', pil_tail(original)), ('abot_pil_after', pil_tail(candidate))]
    shapes = [(1, 48, 96), (1, 64, 80), (3, 48, 96), (3, 64, 80),
              (6, 64, 80), (3, 96, 160), (12, 64, 80), (12, 96, 160)]
    arrays = [np.arange(3 * t * h * w, dtype=np.uint8).reshape(3, t, h, w).transpose(1, 2, 3, 0)
              for t, h, w in shapes]
    for _, convert in variants:
        for frames in arrays:
            for _ in range(2):
                images = convert(frames)
                del images
    for name, convert in variants:
        for frames in arrays:
            t, h, w, _ = frames.shape
            for _ in range(5):
                with perfmark.region(name, pixels=t * h * w, rows=t * h, frames=t):
                    images = convert(frames)
                del images
    print(json.dumps(dict(passed=True, states=shapes, repetitions=5,
                         scope='Actual PIL tail after NumPy astype; excludes GPU math, D2H, NumPy casting, and image destruction.')))
    raise SystemExit(0)
comparisons = 0
matched_unsupported = 0
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    for channels in [3, 4]:
        for count, height, width in [(0, 8, 12), (1, 1, 7), (3, 8, 12), (12, 48, 80),
                                       (3, 63, 65), (3, 64, 64), (12, 96, 160)]:
            planar = torch.randn(channels, count, height, width).to(dtype)
            # Planar, interleaved storage, transposed planes, and spatial slicing.
            interleaved = planar.permute(1, 2, 3, 0).contiguous().permute(3, 0, 1, 2)
            for source in [planar, interleaved, planar.transpose(2, 3), planar[:, :, ::2, ::2]]:
                for resize in [None, (6, 10)] if count else [None]:
                    kw = {} if resize is None else dict(height=resize[0], width=resize[1])
                    try:
                        left = before(None, source, **kw)
                    except NotImplementedError as original_error:
                        # CPU bicubic-antialias resizing does not support FP16.
                        try:
                            after(None, source, **kw)
                        except NotImplementedError as candidate_error:
                            assert candidate_error.args == original_error.args
                        else:
                            raise AssertionError('Candidate changed unsupported-input behavior')
                        matched_unsupported += 1
                        continue
                    right = after(None, source, **kw)
                    assert len(left) == len(right) == count
                    assert [(x.mode, x.size, x.tobytes()) for x in left] == [(x.mode, x.size, x.tobytes()) for x in right]
                    # Returned images retain their contents independently of source mutation.
                    saved = [x.tobytes() for x in right]
                    backup = source.clone()
                    source.zero_()
                    assert saved == [x.tobytes() for x in right]
                    source.copy_(backup)
                    comparisons += 1
print(json.dumps(dict(passed=True, comparisons=comparisons, matched_unsupported=matched_unsupported,
                     scope='Saved patch applied to HEAD; exact PIL mode/size/bytes across RGB/RGBA, dtypes, layouts, resizing, empty sequences, and ownership.'), indent=2))
