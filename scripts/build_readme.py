"""Regenerate README.md from papers.yaml:  python scripts/build_readme.py"""
import os, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "DSamuelHodge/paper2code"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
idx = yaml.safe_load(open(os.path.join(ROOT, "papers.yaml")))
cats, papers = idx["categories"], idx["papers"]

def row(p):
    # arXiv papers link by id; other works (e.g. blog posts) give `url` and `venue`
    src = f"[{p['id']}](https://arxiv.org/abs/{p['id']})" if p.get("arxiv", True) else f"[{p.get('venue', 'link')}]({p['url']})"
    title = f"[{p['title']}]({p['folder']}/) ({p['authors']})"
    smoke = f" · smoke-tested ({p['smoke_test']})" if p.get("smoke_test") else ""
    badge = f"[![Open In Colab]({BADGE})](https://colab.research.google.com/github/{REPO}/blob/main/{p['folder']}/notebook.ipynb)"
    if p.get("official_code"):
        has_nb = os.path.exists(os.path.join(ROOT, p["folder"], "notebook.ipynb"))
        colab = f"[official code]({p['official_code']})" + (f"<br>{badge}" if has_nb else "")
        status = "🔗 official code" + (" + 🧭 guided tour" + smoke if has_nb else "")
    else:
        colab, status = badge, ("📜 patent · " if p.get("type") == "patent" else "") + "📓 notebook" + smoke
    return f"| {title} | {src} | {p['date']} | {p.get('model') or '-'} | {colab} | {status} |"

parts = [open(os.path.join(ROOT, "templates", "README_header.md")).read().rstrip(), "", "## Index", ""]
for key, name in cats.items():
    ps = sorted([p for p in papers if p["category"] == key], key=lambda p: p["id"])
    if not ps: continue
    parts += [f"### {name}", f"`papers/{key}/`", "", "| Paper | arXiv / source | Date | Model | Colab / code | Status |",
              "|---|---|---|---|---|---|", *map(row, ps), ""]
parts.append(open(os.path.join(ROOT, "templates", "README_footer.md")).read().rstrip())
open(os.path.join(ROOT, "README.md"), "w").write("\n".join(parts) + "\n")
print("README.md written:", len(papers), "papers in", len(cats), "categories")
