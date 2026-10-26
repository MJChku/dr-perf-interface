#!/usr/bin/env python3
"""Freeze the archived headers and their quoted FlashInfer dependencies."""
import hashlib,json,pathlib,re,shutil
HERE=pathlib.Path(__file__).resolve().parent
AUDIT=pathlib.Path('/home/ubuntu/gen-glm53-codex-audit')
BASE=pathlib.Path('/home/ubuntu/GX/bw-work/corpus/flashinfer/data/include/flashinfer')
records=[]
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 records.append({'source':str(src),'frozen':str(dst.relative_to(HERE)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
seen=set()
def dep(rel):
 rel=pathlib.Path(rel)
 if rel in seen:return
 seen.add(rel)
 src=BASE/rel;dst=HERE/'frozen/flashinfer'/rel
 if rel==pathlib.Path('comm/trtllm_allreduce_fusion.cuh'):src=AUDIT/'evidence/flashinfer_trtllm_allreduce_fusion.cuh'
 copy(src,dst)
 for name in re.findall(r'^\s*#include\s*"([^"]+)"',src.read_text(),re.M):
  child=(BASE/rel.parent/name).resolve()
  if child.is_file() and child.is_relative_to(BASE):dep(child.relative_to(BASE))
dep('comm/trtllm_allreduce_fusion.cuh')
copy(AUDIT/'src4f5/src/lib_ar_residual_fused.cuh',HERE/'frozen/lib_ar_residual_fused.cuh')
for name in ['lib_allreduce_fusion.cu','lib_allreduce.cuh','decode_ar_norm.cuh']:
 p=AUDIT/'src4f5/src'/name
 records.append({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'use':'read-only protocol/launcher reference'})
(HERE/'sources.json').write_text(json.dumps({'commit':'4f5de3aefec13e115d1a9f7ed0e64f5d1a645695','files':records,'adaptations':['None to frozen headers. Explicit template instantiations in kernels.cu.']},indent=2)+'\n')
print(len(records),'source records')
