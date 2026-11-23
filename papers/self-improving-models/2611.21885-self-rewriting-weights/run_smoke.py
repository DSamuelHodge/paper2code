import os, nbformat, time
from nbclient import NotebookClient
os.environ.update(SKIP_INSTALL="1", MODEL_SIZE="small", MAX_NEW_TOKENS="160", N_DIAG="4", N_PAIRED="4", N_HELDOUT="4",
                  BATCH_SIZE="4", READ_MAX_TOKENS="256", SESSION_PASSES="1", SESSION_TURNS="2", EDITOR_MAX_NEW_TOKENS="300",
                  SESSION_CODE_RUNS="3")
nb = nbformat.read("notebook.ipynb", as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="p2c", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, "notebook_smoke_executed.ipynb")
    print("elapsed", time.time() - t)
