# Web tool for comparing nucleic acid structural parameters

A Python Shiny application that compares sequence-resolved DNA structural coordinates (mean *shape* and
standard deviation) from X-ray, Cryo-EM, MD simulations and the cgNA+ model, at tetramer and dimer level.

Developed as part of the manuscript "Non-local sequence-dependent DNA mechanics across high-resolution
experiments and multi-scale simulations".
Contact: Rahul Sharma, rs25.iitr@gmail.com.
Source: https://github.com/rahul2512/DNA_structure_benchmark (can be downloaded and modified for bespoke analysis).

---

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
shiny run --reload app.py
```

Then open http://localhost:8000. Run the command from inside this folder (the app finds `data/` and
`utils/` relative to `app.py`).

---

## Input file format

Each dataset is a CSV with the sequence in the first column (named `kmer`), an `Info` column in the
second, and one numeric column per structural coordinate. Tetramer and dimer rows can share one file:

```text
kmer,Info,Buckle,Shift,...
AAAA,Shape,0.12,-0.04,...        <- tetramer, mean (shape)
AAAA,Std,0.152,0.94,...          <- tetramer, standard deviation
...
AA,Shape,0.10,-0.03,...          <- dimer, mean (shape)
AA,Std,0.197,0.95,...            <- dimer, standard deviation
```

* The sequence column is `kmer` (the older name `Tetramer` is still read, with a notice).
* `Info` is `Shape` or `Std` (case-insensitive). If the column is missing, all rows are read as Shape.
  Files that still use `Variance` are rejected with a message: take the square root and label the rows `Std`.
* The level of a row follows the length of its sequence: 4 letters = tetramer, 2 letters = dimer
  (letters from A, C, G, T).
* Each (sequence, Info) pair may appear once. 256 tetramers and 16 dimers are expected per Info value;
  fewer rows are accepted with a warning and only shared sequences are compared.
* Coordinate columns are detected automatically and must match between the datasets being compared.
  Missing values (NaN) are allowed. Rows are matched by sequence name, never by row order.

## Default datasets

Six datasets are loaded from `data/` at start-up:

| File | Label in the app |
|------|------------------|
| `xray.csv` | X-ray |
| `cryoem.csv` | Cryo-EM |
| `cryoem_xray.csv` | Cryo-EM + X-ray |
| `md_PS2.csv` | MD (PS2) |
| `md_hexABC.csv` | MD (hexABC) |
| `cgnaplus_PS2.csv` | cgNA+ (PS2) |

**The files shipped here are synthetic placeholders** (generated with fixed seeds by
`python generate_data.py`); `md_hexABC.csv` is the template for the hexABC protocol. Overwrite them with
your real tables (same file names, same format) and restart the app.

File names and labels are defined once, in the `DEFAULT_DATASETS` list near the top of `app.py`. Edit that
list to rename a dataset or add another one; the upload slots, dropdowns, profile selector and About page
follow automatically. Any default can also be replaced for a single session through the upload slots in the
left panel, and one extra custom dataset can be added there.

## Central-step classes

The central dinucleotide of a tetramer is positions 2 and 3 (`CG` in `ACGT`); at dimer level the dimer
itself. With R = purine (A, G) and Y = pyrimidine (C, T) the classes are RR, RY, YR and YY
(64 tetramers or 4 dimers each). Use the **Step class** filter to restrict any plot to one class.

## Metrics

* **Pearson r** - linear association after mean-centering both vectors.
* **Cosine similarity** - angle between the raw (un-centred) vectors; sensitive to the mean level, so it is
  not interchangeable with Pearson r.

## Tabs

| Tab | Content |
|-----|---------|
| Scatter comparison | X vs Y scatter for one coordinate, Pearson r, cosine, regression, difference statistics, table |
| Correlation overview | Pearson r and cosine for every coordinate (bar chart and table) |
| Difference plot | A - B per tetramer or dimer for any two datasets |
| Coordinate profile | Raw values per tetramer or dimer, all datasets overlaid, optional error bars |

The **Info** (Shape / Std), **Level** (Tetramer / Dimer) and **Step class** selectors apply to every tab.

### Coordinate profile: error bars

Tick **Error bars (± k·std)**. The bars are drawn around the Shape values using the `Std` rows of the same
dataset, level and coordinate (no square root is taken); k (default 1) multiplies the standard deviation.
They are only available with Info = Shape. Datasets without `Std` rows are named in a note and drawn
without bars.

### Figure settings per tab

Width, height and tick/legend font size are stored separately for every plot tab:

| Tab | Width x height | Tick / legend font |
|-----|----------------|--------------------|
| Scatter comparison | 720 x 540 | 10 |
| Correlation overview | 720 x 540 | 10 |
| Difference plot | 1200 x 540 | 6 |
| Coordinate profile | 1200 x 540 | 6 |

The **Legend** selector places the legend inside the plot, by default in the emptiest corner; you can also
choose a corner, "Above plot" or "Outside right". Wide figures scroll sideways in the preview; exports
always contain the full figure.

## Export (PNG / PDF / SVG / CSV)

The export buttons save the plot of the **active tab** and its table (CSV), for the current Info, Level and
step class. `requirements.txt` pins `kaleido==0.2.1` and `plotly<7`: Kaleido 1.x (and plotly 7) need a
separate Google Chrome installation and fail without it. If Kaleido cannot run anyway, the app prints a
message in the terminal and draws the image with a built-in matplotlib fallback, so a download is always
a valid PNG/PDF/SVG. For the pixel-exact Plotly rendering run
`pip install "kaleido==0.2.1" "plotly<7"`.

## Project structure

```text
app.py              Shiny application (UI and server)
generate_data.py    Regenerates the synthetic placeholder CSVs in data/
requirements.txt
data/               Default datasets (see above)
utils/metrics.py    Loading, validation, k-mer classification, statistics
utils/export.py     PNG / PDF / SVG export (Kaleido, with matplotlib fallback)
```
