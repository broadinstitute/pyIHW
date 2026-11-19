# Interactive notebook

A [marimo](https://marimo.io) notebook that simulates tests with a known truth,
compares BH with IHW, and lets you explore each of the six QC plots, with
controls for histogram resolution, the decision-boundary axis, and the
rejection cutoff. It runs entirely in your browser (WebAssembly), so the
first load takes a little while as Python and the scientific stack download.

[Open the notebook in a new tab](https://broadinstitute.github.io/pyIHW/app/){ .md-button .md-button--primary }

To run it locally with editable cells, install [Pixi](https://pixi.sh/), then:

```bash
git clone https://github.com/broadinstitute/pyIHW
cd pyIHW
pixi run notebook
```

<iframe src="../app/" width="100%" height="900" style="border: 1px solid #ccc;" loading="lazy"></iframe>
