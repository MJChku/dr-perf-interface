"""unwrap_region.py FILE REGION  -- remove one `with perfmark.region("REGION", ...):`
statement and dedent its body, leaving every other marker in place.

The study measured the output path with the `make_request_output` marker
removed (a Python region costs ~11k instructions to construct, and the fix
skips the call that carried it), so the before/after trees must not count it.
"""
import ast
import re
import sys

path, region = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
pat = re.compile(r'^(\s*)with perfmark\.region\("%s"' % re.escape(region))
out, i, n = [], 0, 0
while i < len(lines):
    m = pat.match(lines[i])
    if not m:
        out.append(lines[i]); i += 1; continue
    indent = len(m.group(1)); i += 1; n += 1
    while i < len(lines) and (lines[i].strip() == "" or len(lines[i]) - len(lines[i].lstrip()) > indent):
        out.append(lines[i][4:] if lines[i].strip() else lines[i]); i += 1
src = "\n".join(out)
ast.parse(src)
open(path, "w").write(src)
print("%s: %d region(s) %r unwrapped" % (path, n, region))
