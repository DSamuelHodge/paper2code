# Paper2Code

Runnable, teaching-first notebooks for machine-learning papers, sorted by topic. Each entry links the paper, says whether official code exists, and, when it does not, ships a Colab notebook that rebuilds the paper's core method with real, executable code. In every notebook, each claim is labelled 📄 *from the paper*, 🧪 *a simplification* or 💭 *an interpretation*, and numbers are quoted only from the PDF.

## The rule

1. **Check for official code first.** Look in the paper, the arXiv comments, the project page, Hugging Face Papers and the authors' GitHub.
2. **Official code exists:** the entry links it (status 🔗), and nothing is reimplemented. Optionally it adds a **guided tour** notebook (🧭) that clones the official repo, runs it, and explains the key equations next to the source that implements them.
3. **No official code:** build a from-scratch teaching notebook (status 📓). If an LLM is needed, it uses a real Qwen ~9B model in 4-bit, or the paper's own model when that is a Qwen of similar size. It always has a small-model switch and falls back to the small model when no GPU is present. It is smoke-tested on CPU before it is pushed. Unofficial reimplementations are linked but not relied on.
4. **No PDFs in the repo.** Link the arXiv page instead, or the project page for works without an arXiv entry.
