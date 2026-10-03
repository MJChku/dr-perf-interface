import ast
from pathlib import Path
specs = [
('source/vllm/renderers/base.py', 'BaseRenderer', '_tokenize_prompt', 'admission.tokenize', "chars=len(prompt['prompt'])"),
('source/vllm/v1/engine/input_processor.py', 'InputProcessor', '_validate_model_input', 'admission.validate', "tokens=len(prompt_input.get('prompt_token_ids') or [])"),
('source/vllm/v1/engine/output_processor.py', 'OutputProcessor', 'add_request', 'admission.output_init', "tokens=len(request.prompt_token_ids or [])"),
('source/vllm/v1/engine/core.py', 'EngineCore', 'preprocess_add_request', 'admission.core_init', "tokens=len(request.prompt_token_ids or [])"),
('source/vllm/v1/engine/llm_engine.py', 'LLMEngine', 'add_request', 'admission.request', "tokens=len(prompt.get('prompt_token_ids') or []) if isinstance(prompt, dict) else 0, outputs=params.n if isinstance(params, SamplingParams) else 1, max_tokens=(params.max_tokens or 0) if isinstance(params, SamplingParams) else 0, resident=len(self.output_processor.request_states)"),
('source/vllm/v1/core/kv_cache_utils.py', None, 'request_block_hasher', 'admission.hash_blocks', "new_blocks=max(0, request.num_tokens // hash_block_size - len(request.block_hashes)), hashed_tokens=max(0, request.num_tokens // hash_block_size - len(request.block_hashes)) * hash_block_size")]
for file, cls, fn, region, pcvs in specs:
    p=Path(file); src=p.read_text(); tree=ast.parse(src)
    scope=next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name==cls), tree) if cls else tree
    node=next(n for n in ast.walk(scope) if isinstance(n,ast.FunctionDef) and n.name==fn)
    first=node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value,str): first=node.body[1]
    lines=src.splitlines(True); start=first.lineno-1; end=node.end_lineno
    indent=lines[start][:len(lines[start])-len(lines[start].lstrip())]
    lines[start:end]=[indent+f'with _admission_region("{region}", {pcvs}):\n']+['    '+line if line.strip() else line for line in lines[start:end]]
    lines.insert(2,'from perfmark import region as _admission_region\n')
    edited=''.join(lines); ast.parse(edited); p.write_text(edited)
