## Layout

```
papers/
  <category>/                     # kebab-case topic, created only when needed
    <arxiv-id>-<short-slug>/      # e.g. 2611.26450-cpu-inside-an-llm
      README.md                   # badge, paper links, code status, contents   (templates/paper_README.md)
      notebook.ipynb              # the Colab notebook (generated)
      build_notebook.py           # regenerates notebook.ipynb                  (templates/build_notebook_template.py)
      run_smoke.py                # CPU smoke test with the small model
      notebook_smoke_executed.ipynb
papers.yaml                       # the index: one entry per paper
scripts/build_readme.py           # regenerates this README from papers.yaml
templates/                        # templates for a new entry and for this README
```

## Contributing a paper

1. Check for official code (see the rule above). If it exists, add a `papers.yaml` entry with `official_code: <url>` and `status: official-code`, and stop.
2. Otherwise, pick or create a category, then copy `templates/paper_README.md` and `templates/build_notebook_template.py` into `papers/<category>/<arxiv-id>-<slug>/`.
3. Write the notebook in the template's order:
   - a plain-English overview
   - each key equation in LaTeX, followed by the code that implements it
   - a faithful minimal reimplementation of the main method
   - a toy experiment
   - limitations and exercises
   - an appendix of the paper's numbers, with table or figure references
4. Tag every claim 📄 / 🧪 / 💭. Never invent results.
5. Run `python build_notebook.py`, then `python run_smoke.py` on CPU with the small model. Commit both notebooks.
6. Add the entry to `papers.yaml` and run `python scripts/build_readme.py`. Check that the Colab badge opens.
