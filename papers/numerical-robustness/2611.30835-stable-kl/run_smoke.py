import os, nbformat, time
from nbclient import NotebookClient
# CPU smoke defaults: one distillation seed at V=32768, 2 x 512 WikiText-2 tokens with the paper's Qwen2.5-1.5B-Instruct
os.environ.setdefault("N_WINDOWS", "2")
nb = nbformat.read("notebook.ipynb", as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="p2c", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, "notebook_smoke_executed.ipynb")
    print("elapsed", time.time() - t)
