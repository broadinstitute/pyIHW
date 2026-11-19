# Interactive notebook

A [marimo](https://marimo.io) notebook that simulates tests with a known truth,
compares BH with IHW, and walks through the QC plots. It runs entirely in your
browser (WebAssembly), so the first load takes a little while as Python and
the scientific stack download.

[Open the notebook in a new tab](https://broadinstitute.github.io/pyIHW/app/){ .md-button .md-button--primary }

To run it locally with editable cells:

```bash
git clone https://github.com/broadinstitute/pyIHW
cd pyIHW
uvx marimo edit --sandbox examples/ihw_toy_example.py
```

<iframe src="../app/" width="100%" height="900" style="border: 1px solid #ccc;" loading="lazy"></iframe>
