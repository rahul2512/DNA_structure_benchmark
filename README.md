# Web tool for comparing nucleic acid structural parameters

A Python Shiny application for comparing sequence-resolved DNA structural
coordinates across X-ray crystallography, molecular dynamics (MD) simulation,
and optionally coarse-grained DNA (cgDNA) datasets.

---

## Scientific purpose

DNA structural parameters — such as Twist, Roll, Tilt, Slide, Propeller, or
Buckle — vary depending on the nucleotide sequence context.  The finest
practically useful resolution is the DNA **tetramer**: the 4-base subsequence
centred on the dinucleotide step of interest.  All 4^4 = 256 possible tetramers
can be assigned coordinates from different experimental or computational sources.

This application enables side-by-side comparison of those 256 tetramer
coordinates from up to six datasets (five defaults plus one custom upload), using:

- Pearson correlation to assess linear agreement
- Cosine similarity to measure vector orientation
- Regression analysis, filtering by central-step class, heatmaps, and more

---

## Input format

Each dataset must be a plain CSV file with the following structure:

| Tetramer | Coord_1 | Coord_2 | … | Coord_18 |
|----------|---------|---------|---|----------|
| AAAA     | 0.123   | 35.4    | … | 3.38     |
| AAAC     | 0.087   | 34.8    | … | 3.35     |
| …        | …       | …       | … | …        |
| TTTT     | −0.456  | 36.1    | … | 3.41     |

Requirements:
- The **first column must be named `Tetramer`**.
- Every tetramer must be exactly 4 characters from the set A, C, G, T.
- The recommended number of tetramers is 256 (all possible 4-mers).
- Remaining columns must be numeric. Non-numeric columns are dropped automatically.
- Column names beyond `Tetramer` are detected automatically — there is no
  requirement for any specific naming convention, though names containing
  `intra`/`inter` (or common parameter keywords such as `roll`, `twist`,
  `buckle`) are automatically labelled in the UI.

---

## Central-step classification

The **central dinucleotide** of a tetramer is formed by positions 2 and 3
(1-indexed).  For example, in `ACGT`, the central dinucleotide is `CG`.

Each base is classified as a purine (A or G) or a pyrimidine (C or T),
giving four **step classes**:

| Class | Meaning              | Example central steps |
|-------|----------------------|-----------------------|
| RR    | Purine / Purine      | AA, AG, GA, GG        |
| RY    | Purine / Pyrimidine  | AC, AT, GC, GT        |
| YR    | Pyrimidine / Purine  | CA, CG, TA, TG        |
| YY    | Pyrimidine / Pyrimidine | CC, CT, TC, TT     |

Each class therefore contains 4 × 4 × 4 = 64 tetramers (4 choices of
flanking base-pair × 4 choices on each side × 16 dinucleotides divided into
four equal groups of four).

---

## Metrics

### Pearson correlation (r)

Pearson r measures **linear association** after centering both vectors:

```
r = cov(x, y) / (σ_x · σ_y)
```

It is insensitive to additive offsets or multiplicative scaling, making it
appropriate for comparing structural parameter trends across datasets that
may have systematic biases.

### Cosine similarity

Cosine similarity measures the **angle** between the two raw (un-centred)
vectors:

```
cos(θ) = (x · y) / (||x|| · ||y||)
```

Unlike Pearson r, cosine similarity is sensitive to the mean level of both
vectors.  A high cosine with a low Pearson r indicates good shape agreement
but a mean-level difference (systematic bias).

**These metrics are not interchangeable.**  Inspecting both simultaneously
(see the *Similarity analysis* tab) provides a more complete picture.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running locally

```bash
shiny run --reload app.py
```

Then open `http://localhost:8000` in a browser.

---

## Default datasets

Five placeholder datasets are included in `data/`. Each has 256 tetramer and 16 dimer rows for both
`Shape` and `Variance`, over 18 structural coordinates:

| File | Description |
|------|-------------|
| `xray.csv` | X-ray reference set with small Gaussian noise |
| `cryoem.csv` | Cryo-EM: scale + bias + noise |
| `cryoem_xray.csv` | Cryo-EM + X-ray: average of the two shape sets, reduced variance |
| `md.csv` | MD: slightly scaled + biased + moderate noise |
| `cgnaplus.csv` | cgNA+: different scale / bias / noise |

They are synthetic and generated with fixed random seeds by `python generate_data.py`.
**Replace them with your real data before publishing** (see "Default data files and how to replace them" below).

---

## Application tabs

| Tab                  | Content                                                    |
|----------------------|------------------------------------------------------------|
| Scatter comparison   | Main X vs Y scatter plot for one coordinate                |
| Correlation overview  | Pearson r and cosine for all 18 coordinates (bar chart + table) |
| Difference plot      | Bar chart of A − B per tetramer or dimer for any two datasets |
| Coordinate profile   | Raw values per tetramer or dimer, all datasets overlaid, optional error bars |

---

## Export functionality

| Format | Contents                                   |
|--------|--------------------------------------------|
| PNG    | Current scatter plot at 2× pixel density  |
| PDF    | Vector PDF of the current scatter plot     |
| SVG    | Vector SVG of the current scatter plot     |
| CSV    | Currently filtered comparison data table  |

PNG/PDF/SVG export uses [Kaleido](https://github.com/plotly/Kaleido) for
high-fidelity static rendering of Plotly figures.

---

## Interpretation guide

1. **Start** on the *Scatter comparison* tab with your X-ray and MD datasets.
2. Use the **Coordinate** dropdown to inspect each of the 18 parameters.
3. Use **Central-step filter** (RR/RY/YR/YY) to check whether agreement is
   step-class-dependent.
4. Switch to **Correlation overview** for a global r profile across all 18 coords.
5. Use **Similarity analysis** to identify coordinates where Pearson r and
   cosine diverge (indicative of systematic offsets).
6. Use **Correlation heatmap** to see which step classes drive agreement or
   disagreement for each parameter.
7. Use **Difference plot** to identify individual tetramers with unusually
   large discrepancies.

---

## Project structure

```
tetramer_comparison/
├── app.py              Main Shiny application
├── requirements.txt    Python dependencies
├── README.md           This file
├── generate_data.py    Script to regenerate synthetic datasets
├── data/
│   ├── xray.csv
│   ├── md.csv
│   └── cgnaplus.csv
└── utils/
    ├── __init__.py
    └── metrics.py      Pure-Python analysis functions (no UI)
```

---

## Dependencies

- **shiny / shinywidgets** — reactive UI framework for Python
- **plotly** — interactive publication-quality figures
- **pandas / numpy** — data handling
- **scipy** — Pearson correlation
- **kaleido** — static image export from Plotly


## Citation, contact and source

Developed as part of the manuscript "Non-local sequence-dependent DNA mechanics across
high-resolution experiments and multi-scale simulations".

Contact: Rahul Sharma, rs25.iitr@gmail.com

The code can be downloaded and modified for bespoke analysis.

Note: the CSVs in `data/` are synthetic placeholders generated by `generate_data.py`.
Replace them with the real X-ray, Cryo-EM, MD and cgNA+ tables before publishing.

## Input file format: Info, tetramers and dimers in one file

Each dataset is a CSV with the sequence in the first column, an `Info` column in the second,
and one numeric column per structural coordinate:

```text
Tetramer,Info,Buckle,Shift,...
AAAA,Shape,0.12,-0.04,...        <- tetramer, mean (shape)
AAAA,Variance,0.041,0.88,...      <- tetramer, variance
...
AA,Shape,0.10,-0.03,...           <- dimer, mean (shape)
AA,Variance,0.039,0.91,...        <- dimer, variance
```

* `Info` is `Shape` or `Variance` (case-insensitive). If the column is missing, all rows are read as Shape.
* The level of a row follows the length of its sequence: 4 letters = tetramer, 2 letters = dimer.
* Each (sequence, Info) pair may appear once. 256 tetramers and 16 dimers are expected per Info value.
* Missing values (NaN) are allowed. Rows are matched by sequence name, never by row order.

In the app, the **Info** (Shape / Variance), **Level** (Tetramer / Dimer) and **Step class**
(RR/RY/YR/YY, R = purine, Y = pyrimidine) selectors in the control bar apply to every tab.
The five default files in `data/` follow this format and can be downloaded from the About tab.

## Export (PNG / PDF / SVG / CSV)

The export buttons save the plot on the **active tab** and its table (CSV), using the current
Info, Level and step-class selection. `requirements.txt` pins `kaleido==0.2.1` and `plotly<7`:
Kaleido 1.x (and plotly 7) need a separate Google Chrome installation and fail without it.
If Kaleido cannot run anyway, the app prints a message in the terminal and renders the image with
a built-in matplotlib fallback, so the downloaded file is always a valid PNG/PDF/SVG. To get the
pixel-exact Plotly rendering run:

```bash
pip install "kaleido==0.2.1" "plotly<7"
```

## Coordinate profile: error bars

Tick **Error bars (± k·√variance)** on the Coordinate profile tab. The bars are drawn around the
Shape values using the Variance rows of the same dataset, level and coordinate; k (default 1) is a
multiplier of the standard deviation. They are only available with Info = Shape. Datasets that have
no Variance rows are listed in a note and drawn without bars. The CSV export of the tab contains
the plotted value and the error size for each dataset.

## Default data files and how to replace them

The app loads five files from `data/` at start-up: `xray.csv`, `cryoem.csv`, `cryoem_xray.csv`
(Cryo-EM + X-ray), `md.csv` and `cgnaplus.csv`. They are placeholders generated by
`generate_data.py`: overwrite them with your real tables (same file names, same format, see above)
and restart the app. Each dataset can also be replaced for one session through the upload slots in the
left panel, and a sixth, custom dataset can be added there.

## Figure settings per tab

Width, height and tick/legend font size are stored separately for every plot tab:

| Tab | Width x height | Tick / legend font |
|-----|----------------|--------------------|
| Scatter comparison | 720 x 540 | 10 |
| Correlation overview | 720 x 540 | 10 |
| Difference plot | 1200 x 540 | 6 |
| Coordinate profile | 1200 x 540 | 6 |

Changing a value affects only the tab you are on; the value is restored when you come back to that tab.
The **Legend** selector places the legend inside the plot, by default in the emptiest corner
(`Auto (best fit)`); you can also choose a corner, `Above plot` or `Outside right`.
Wide figures scroll sideways in the preview; exports always contain the full figure.

Source code: https://github.com/rahul2512/DNA_structure_benchmark
