import nbformat, time
from nbclient import NotebookClient
nb = nbformat.read("notebook.ipynb", as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="p2c", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, "notebook_smoke_executed.ipynb")
    print("elapsed", time.time() - t)
