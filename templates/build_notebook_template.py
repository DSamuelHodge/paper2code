"""Skeleton generator for a Paper2Code notebook. Copy into papers/<category>/<id>-<slug>/ and fill in."""
import os, nbformat as nbf
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
COLAB = "https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/<category>/<id>-<slug>/notebook.ipynb"

md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

# Paper2Code: *<Title>*

**Paper:** <authors>. arXiv:<id>. [abs](https://arxiv.org/abs/<id>) · [pdf](https://arxiv.org/pdf/<id>)

**Is there official code?** <answer, and where you looked>

**How to read the labels**
- 📄 **Paper:** stated in the PDF (numbers quoted, with table or figure).
- 🧪 **Simplification:** a choice this notebook makes; not the paper's setup.
- 💭 **Interpretation:** my reading where the PDF leaves details open.
""")
md("## Contents\n0. Setup\n1. The idea in plain English\n2. Key equations, each followed by code\n3. Main method\n4. Toy experiment\n5. Limitations\n6. Exercises\n7. Appendix: numbers from the paper")
code(r"""
import os, sys, subprocess
if os.environ.get("SKIP_INSTALL") != "1":
    pkgs = ["transformers>=5.24", "accelerate"]
    import torch
    if torch.cuda.is_available(): pkgs.append("bitsandbytes>=0.45")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)
""")
code(r"""
import os, torch
MODEL_SIZE = os.environ.get("MODEL_SIZE", "9B")      # "9B" (4-bit, GPU) or "small" (CPU)
MODEL_IDS = {"9B": "Qwen/Qwen3.5-9B", "small": "Qwen/Qwen3.5-0.8B"}
HAS_GPU = torch.cuda.is_available()
if MODEL_SIZE != "small" and not HAS_GPU:
    print("No GPU found -> switching to the small model."); MODEL_SIZE = "small"
MODEL_ID = MODEL_IDS[MODEL_SIZE]; LOAD_4BIT = MODEL_SIZE == "9B" and HAS_GPU
""")
md("## 1. The idea in plain English\n<...>")
md(r"## 2. Key equations\n$$<equation>$$\n📄 <where it is in the paper>")
code("# code that implements the equation above")
md("## 3. Main method\n<...>")
md("## 4. Toy experiment\n<...>")
md("## 5. Limitations\n- 🧪 <...>")
md("## 6. Exercises\n1. <...>")
md("## 7. Appendix: numbers from the paper\n| claim | value | where |\n|---|---|---|")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"}
nbf.validate(nb)
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook.ipynb"))
