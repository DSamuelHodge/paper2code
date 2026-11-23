import os, nbformat, time
from nbclient import NotebookClient
os.environ.update(N_TEST_MAZES=os.environ.get("N_TEST_MAZES", "1"), MAZE_SIDE="15", MAX_PARITY_LEN="256", HANOI_DISKS="10")
nb = nbformat.read("notebook.ipynb", as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="p2c", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, "notebook_smoke_executed.ipynb")
    print("elapsed", time.time() - t)
