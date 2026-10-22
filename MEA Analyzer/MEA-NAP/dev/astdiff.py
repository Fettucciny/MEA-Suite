"""Which top-level functions / classes / assignments differ between two
revisions of a module, by AST hash (whitespace and comments ignored).

    python dev\astdiff.py OLD.py NEW.py

Use it to prove a plotting-only change left every numerical routine untouched:
the CHANGED list should contain nothing you did not mean to change.
"""
import ast, hashlib, sys
def items(path):
    t = ast.parse(open(path, encoding="utf-8").read()); out = {}
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            k = n.name
        elif isinstance(n, ast.Assign):
            nm = [x.id for x in n.targets if isinstance(x, ast.Name)]
            if not nm: continue
            k = "=" + nm[0]
        else: continue
        out[k] = hashlib.sha256(ast.dump(n).encode()).hexdigest()[:12]
    return out
a, b = items(sys.argv[1]), items(sys.argv[2])
for lbl, ks in (("IDENTICAL", sorted(k for k in a if k in b and a[k]==b[k])),
                ("CHANGED", sorted(k for k in a if k in b and a[k]!=b[k])),
                ("REMOVED", sorted(k for k in a if k not in b)),
                ("ADDED", sorted(k for k in b if k not in a))):
    print(f"{lbl} ({len(ks)}): {', '.join(ks)}\n")
