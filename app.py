"""
Tetramer Structural Comparison — Shiny for Python Application
=============================================================
Compares sequence-resolved DNA structural coordinates across
X-ray crystallography, molecular dynamics (MD) simulation,
Cryo-EM, and coarse-grained (cgNA+) datasets.

The built-in datasets (data/example_xray.csv, example_cryoem.csv,
data/example_md.csv, example_cgnaplus.csv) are loaded
automatically on startup so the app is immediately usable when
deployed.  Users can:
  • Replace any built-in dataset by uploading their own CSV in the
    corresponding slot.
  • Add a fourth custom dataset (any name) via the "Custom dataset"
    upload slot.

Entry point:  shiny run --reload app.py
"""

from __future__ import annotations

import os
import io
from itertools import product as iproduct
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from utils.export import ExportError, figure_to_bytes

from utils.metrics import (
    CLASS_COLOURS,
    STEP_CLASS_LABELS,
    calculate_all_coords_stats,
    calculate_cosine,
    calculate_pearson,
    calculate_rmse,
    calculate_summary_statistics,
    check_dataset_compatibility,
    classify_central_step,
    detect_coord_type,
    get_coord_type_label,
    linear_regression,
    load_dataset,
    prepare_comparison,
    describe_dataset,
    select_view,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Built-in default datasets — loaded once at module import time.
# Paths are relative to this file so they work regardless of cwd.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "data")

def _load_builtin(filename: str, label: str):
    """Load one of the bundled example CSVs.  Returns (df, label) or (None, label)."""
    path = os.path.join(_DATA_DIR, filename)
    if not os.path.exists(path):
        return None, label
    df_raw, err = load_dataset(path)
    if err:
        return None, label
    df, warns = validate_dataset(df_raw, label)
    return df, label   # df may be None if validation failed, that's fine

BUILTIN_DATASETS = {
    "xray":   _load_builtin("example_xray.csv",     "X-ray"),
    "cryoem": _load_builtin("example_cryoem.csv",   "Cryo-EM"),
    "md":     _load_builtin("example_md.csv",        "MD"),
    "cgdna":  _load_builtin("example_cgnaplus.csv",  "cgNA+"),
}
# BUILTIN_DATASETS[key] = (df_or_None, label_str)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-dataset marker symbols and colours for the profile plot
DATASET_MARKERS = {
    "xray":   dict(symbol="square",          color="#1a2744", size=6),
    "md":     dict(symbol="diamond",         color="#e63946", size=7),
    "cryoem": dict(symbol="hexagon",         color="#e9a100", size=7),
    "cgdna":  dict(symbol="circle",          color="#2a9d8f", size=7),
    "custom": dict(symbol="triangle-up",     color="#9b59b6", size=7),
}

DEFAULT_FIG_W  = 760
DEFAULT_FIG_H  = 540

PLOT_LAYOUT_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    # NOTE: `font` is intentionally omitted here so each call can pass it
    # as an explicit keyword without a duplicate-key conflict.
    margin=dict(l=72, r=40, t=58, b=70),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#e8e8e8",
        linecolor="#aaaaaa", linewidth=1,
        ticks="outside", ticklen=4, zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e8e8e8",
        linecolor="#aaaaaa", linewidth=1,
        ticks="outside", ticklen=4, zeroline=False,
    ),
)

# Base font dict – family + colour stay constant; size is overridden per call
_FONT_BASE = dict(family="Arial, Helvetica, sans-serif", color="#222222")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = ui.HTML("""
<style>
/* ── Global ── */
body { font-family: 'Arial', sans-serif; background: #f0f2f7; margin:0; }

/* ── Header ── */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, #0f1f3d 0%, #1a3a6b 60%, #1e4d8c 100%);
  color: white;
  padding: 0 28px;
  height: 56px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.app-header-left  { display:flex; align-items:center; gap:12px; }
.app-header-icon  {
  font-size: 1.5rem; line-height:1;
  background: rgba(255,255,255,0.12);
  border-radius: 8px;
  padding: 5px 8px;
}
.app-header-title { font-size:1.1rem; font-weight:700; letter-spacing:0.02em; }
.app-header-sub   { font-size:0.72rem; color:#90b4e0; margin-top:1px; }
.app-header-right { display:flex; align-items:center; gap:10px; }
.app-header-pill  {
  background: rgba(255,255,255,0.13);
  border: 1px solid rgba(255,255,255,0.22);
  color: #c5d9f0;
  font-size: 0.68rem;
  font-weight:600;
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.04em;
}
.app-header-author {
  font-size:0.72rem; color:#90b4e0; font-style:italic;
}

/* ── Control bar ── */
.ctrl-bar {
  background: white;
  border: 1px solid #dde2ec;
  border-radius: 8px;
  padding: 8px 10px 6px 10px;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.ctrl-group-label {
  font-size: 0.63rem; font-weight:700; color:#9ca3af;
  text-transform: uppercase; letter-spacing:0.08em;
  margin-bottom: 3px; white-space: nowrap;
}
/* Compact all inputs inside the ctrl-bar */
.ctrl-bar .form-group  { margin-bottom: 3px !important; }
.ctrl-bar label        { font-size: 0.76rem !important; color:#374151;
                          margin-bottom: 1px !important; white-space:nowrap; overflow:hidden;
                          text-overflow:ellipsis; }
.ctrl-bar .form-control{ font-size:0.8rem !important; padding:2px 5px !important;
                          height:26px !important; }
.ctrl-bar input[type=number]  { height:26px !important; padding:2px 5px !important; }
.ctrl-bar input[type=text]    { height:26px !important; padding:2px 5px !important;
                                  font-size:0.8rem !important; }
.ctrl-bar .checkbox label { font-size:0.76rem !important; }
.ctrl-bar .bslib-col      { padding-left:4px !important; padding-right:4px !important; }
/* Export buttons */
.ctrl-bar .btn { font-size:0.76rem !important; padding:3px 7px !important; }

/* ── KPI cards ── */
.kpi-card {
  background:white; border-radius:7px; border:1px solid #dde2ec;
  padding:8px 14px 6px 14px; text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,0.06); min-width:105px;
}
.kpi-label { font-size:0.67rem; color:#6b7280; text-transform:uppercase;
             letter-spacing:0.06em; margin-bottom:1px; }
.kpi-value { font-size:1.45rem; font-weight:700; color:#1a2744; line-height:1.1; }
.kpi-sub   { font-size:0.65rem; color:#9ca3af; margin-top:1px; }

/* ── Sidebar section titles ── */
.sb-title {
  font-size:0.7rem; font-weight:700; color:#6b7280;
  text-transform:uppercase; letter-spacing:0.07em;
  margin:12px 0 4px 0; border-top:1px solid #e5e7eb; padding-top:10px;
}
.sb-title:first-child { border-top:none; margin-top:0; padding-top:0; }

/* ── Message boxes ── */
.info-box {
  background:#f0f4ff; border-left:3px solid #3b5bdb;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px; margin-bottom:5px;
}
.warn-box {
  background:#fff7ed; border-left:3px solid #f59e0b;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px;
}
.err-box {
  background:#fef2f2; border-left:3px solid #ef4444;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px;
}
.ok-box {
  background:#f0fdf4; border-left:3px solid #22c55e;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:3px;
}
.stat-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }

/* ── Custom upload block ── */
.custom-upload-block {
  background:#fafafa; border:1px dashed #c7cfe0;
  border-radius:6px; padding:7px 9px 5px 9px; margin-top:4px;
}
.custom-upload-block .sb-title { border-top:none; margin-top:0; padding-top:0; }

/* ── Export buttons ── */
.export-btn-col .btn { width:100%; margin-bottom:3px; font-size:0.78rem; padding:4px 8px; }

/* ── Description / About page ── */
.desc-page { max-width:840px; margin:0 auto; padding:10px 6px 50px 6px; }
.desc-page h3 { color:#1a2744; margin-top:1.6rem; margin-bottom:0.35rem; font-size:1.05rem; }
.desc-page h4 { color:#374151; margin-top:1.1rem; margin-bottom:0.25rem; font-size:0.95rem; }
.desc-page p, .desc-page li { font-size:0.9rem; color:#374151; line-height:1.65; }
.desc-page code { background:#f1f5f9; padding:1px 5px; border-radius:3px;
                   font-size:0.85rem; color:#1a2744; }
.desc-page table { border-collapse:collapse; width:100%; margin:8px 0; font-size:0.87rem; }
.desc-page th { background:#1a2744; color:white; padding:6px 10px; text-align:left; }
.desc-page td { padding:5px 10px; border-bottom:1px solid #e5e7eb; }
.desc-page tr:nth-child(even) td { background:#f8fafc; }
.credit-footer { margin-top:2.5rem; padding-top:0.9rem; border-top:1px solid #e5e7eb;
                 font-size:0.8rem; color:#9ca3af; text-align:right; }
</style>
""")

# ---------------------------------------------------------------------------
# Helper: KPI card HTML
# ---------------------------------------------------------------------------

def _kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    CUSTOM_CSS,

    # ── Compact header ──
    ui.div(
        {"class": "app-header"},
        ui.div(
            ui.HTML('<div class="app-header-title">Web tool for comparing nucleic acid structural parameters</div>'),
            ui.HTML('<div class="app-header-sub">Sequence-resolved DNA structural coordinates — X-ray · Cryo-EM · MD · cgNA+</div>'),
        ),
    ),

    # ============================================================
    # TOP-LEVEL NAV — Description tab sits outside the sidebar layout
    # ============================================================
    ui.navset_tab(

        # ---- Description tab (full width, no sidebars) ----
        ui.nav_panel(
            "About",
            ui.div(
                {"class": "desc-page"},
                ui.HTML("""
<h3>Purpose</h3>
<p>This web tool compares <strong>sequence-resolved nucleic acid structural parameters</strong>
across experimental and computational sources: X-ray crystallography, Cryo-EM, molecular
dynamics (MD) simulations and the coarse-grained cgNA+ model. Structural parameters such as
Twist, Roll, Propeller or Rise depend on the local sequence context; here they are resolved at the
level of the DNA <strong>tetramer</strong> (the four-base context centred on the dinucleotide step
of interest), covering all 4<sup>4</sup> = 256 tetramers simultaneously.</p>

<h3>Default datasets</h3>
<p>Default DNA structural coordinates for <strong>X-ray, Cryo-EM, MD and cgNA+</strong> are loaded
automatically on start-up from the <code>data/</code> folder
(<code>example_xray.csv</code>, <code>example_cryoem.csv</code>, <code>example_md.csv</code>,
<code>example_cgnaplus.csv</code>). Any of them can be replaced by uploading your own CSV, and an
additional custom dataset can be added alongside them.</p>

<h3>Data information</h3>
<p><strong>Info = Shape</strong> holds the mean (ground-state) value of each structural coordinate;
<strong>Info = Variance</strong> holds the variance of the same coordinate (the diagonal of its covariance).
Both are available at the <strong>tetramer</strong> level (256 sequences) and at the <strong>dimer</strong>
level (16 sequences), and the <em>Info</em> and <em>Level</em> selectors in the control bar switch every
plot and table between them.</p>
<h4>cgNA+</h4>
<p>cgNA+ is a sequence-dependent coarse-grained model of double-stranded nucleic acids. Each
configuration is described by internal coordinates of rigid bases and phosphate groups, and the
model is a Gaussian (multivariate normal) distribution over these coordinates whose parameters are
fitted to all-atom molecular dynamics data. For any sequence it therefore returns a ground-state
shape (the mean) and a stiffness matrix whose inverse is the covariance; the stiffness couples
neighbouring steps, which is how it captures the non-local sequence dependence of DNA mechanics. The
cgNA+ values used here are the mean (Shape) and the variance (Variance) of the coordinates per
tetramer, with dimer values for the central step.</p>

<h3>Input file format for custom comparisons</h3>
<p>Each dataset is a plain <code>.csv</code> file. Besides the sequence and the coordinates, it has an
<code>Info</code> column that says whether a row holds the <em>Shape</em> or the <em>Variance</em>:</p>
<table>
  <tr><th>Tetramer</th><th>Info</th><th>Coord_1</th><th>Coord_2</th><th>…</th><th>Coord_18</th></tr>
  <tr><td>AAAA</td><td>Shape</td><td>0.123</td><td>35.4</td><td>…</td><td>3.38</td></tr>
  <tr><td>AAAC</td><td>Shape</td><td>0.087</td><td>34.8</td><td>…</td><td>3.35</td></tr>
  <tr><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td></tr>
  <tr><td>AAAA</td><td>Variance</td><td>0.041</td><td>0.88</td><td>…</td><td>0.002</td></tr>
  <tr><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td></tr>
  <tr><td>AA</td><td>Shape</td><td>0.101</td><td>35.0</td><td>…</td><td>3.36</td></tr>
  <tr><td>AA</td><td>Variance</td><td>0.039</td><td>0.91</td><td>…</td><td>0.002</td></tr>
</table>
<ul>
  <li>The <strong>first column must be named <code>Tetramer</code></strong> and the second <code>Info</code>. <code>Info</code> is either <code>Shape</code> or <code>Variance</code>; if the column is missing, every row is read as Shape.</li>
  <li>The level of a row is set by the length of its sequence: <strong>4 letters = tetramer</strong>, <strong>2 letters = dimer</strong> (letters from <code>A</code>, <code>C</code>, <code>G</code>, <code>T</code>). Tetramer, dimer, Shape and Variance rows can all share one file, using the same coordinate columns.</li>
  <li>A sequence may appear once per <code>Info</code> value; the same sequence and Info twice is rejected.</li>
  <li>Per Info value, 256 tetramer rows and 16 dimer rows are expected. Fewer rows are accepted with a warning and only shared sequences are compared. A file with no rows for the selected Info or level is left out of that view.</li>
  <li>All other columns must be numeric; names are detected automatically. Missing values (NaN) are allowed and excluded pairwise.</li>
  <li>Coordinate names must match between the datasets being compared. Names containing keywords such as <em>buckle</em>, <em>propeller</em>, <em>twist</em>, <em>roll</em> are labelled Intra or Inter.</li>
  <li>Rows are always matched by sequence name, never by row order.</li>
</ul>
<p><strong>Download example files</strong> (each has 256 tetramer + 16 dimer rows for both Shape and Variance):</p>
<ul>
  <li><a href="example_xray.csv" download>example_xray.csv</a></li>
  <li><a href="example_cryoem.csv" download>example_cryoem.csv</a></li>
  <li><a href="example_md.csv" download>example_md.csv</a></li>
  <li><a href="example_cgnaplus.csv" download>example_cgnaplus.csv</a></li>
</ul>

<h3>Central-step classification</h3>
<p>The central dinucleotide of a tetramer is positions 2 and 3 (e.g. <code>CG</code> in <code>ACGT</code>); at the dimer level the dimer itself is the dinucleotide.
Each base is a purine (A, G) or pyrimidine (C, T), giving four classes of 64 tetramers each:</p>
<table>
  <tr><th>Class</th><th>Meaning</th><th>Example central steps</th></tr>
  <tr><td>RR</td><td>Purine / Purine</td><td>AA, AG, GA, GG</td></tr>
  <tr><td>RY</td><td>Purine / Pyrimidine</td><td>AC, AT, GC, GT</td></tr>
  <tr><td>YR</td><td>Pyrimidine / Purine</td><td>CA, CG, TA, TG</td></tr>
  <tr><td>YY</td><td>Pyrimidine / Pyrimidine</td><td>CC, CT, TC, TT</td></tr>
</table>
<p>Use the <em>Filters</em> in the control bar to choose the <strong>Info</strong> (Shape or Variance), the <strong>Level</strong> (tetramer or dimer) and, optionally, one step class to restrict any plot to.</p>

<h3>Statistical metrics</h3>
<h4>Pearson correlation (r)</h4>
<p>Linear association after mean-centering both vectors; insensitive to additive offsets and
multiplicative scaling.</p>
<h4>Cosine similarity</h4>
<p>Angle between the raw (un-centred) vectors, <code>cos θ = (x · y) / (‖x‖ ‖y‖)</code>. Unlike
Pearson r it is sensitive to the mean level, so the two metrics are <em>not</em> interchangeable:
high cosine with low Pearson r indicates similar shape but a systematic offset.</p>

<h3>Analysis tabs</h3>
<table>
  <tr><th>Tab</th><th>Description</th></tr>
  <tr><td>Scatter comparison</td><td>X vs Y scatter for one coordinate; Pearson r, cosine, regression, difference statistics and data table</td></tr>
  <tr><td>Coordinate overview</td><td>Pearson r and cosine for every coordinate (bar chart and table), for the selected Info and level</td></tr>
  <tr><td>Difference plot</td><td>A − B per tetramer or dimer for any two chosen datasets, sortable alphabetically or by absolute difference</td></tr>
  <tr><td>Coordinate profile</td><td>Raw coordinate values (shape or variance) per tetramer or dimer with all selected datasets overlaid; optional error bars of &plusmn;k&middot;&radic;variance around the Shape values</td></tr>
</table>

<h3>Export</h3>
<p>The export buttons in the control bar save <strong>the plot of the tab you are on</strong> (scatter,
coordinate overview, difference or coordinate profile) as PNG, PDF or SVG, and its table as CSV, for the
selected Info, level and step class. Figure width, height, fonts, title, axis labels (scatter plot) and
file name can be set beforehand. Images are rendered with
<a href="https://github.com/plotly/Kaleido" target="_blank">Kaleido</a> (<code>kaleido==0.2.1</code>, no
Chrome needed); if Kaleido cannot run, a built-in matplotlib renderer draws the same plot, so a download
is always a valid image.</p>

<h3>Citation</h3>
<p>This work has been developed as part of the manuscript
<em>&ldquo;Non-local sequence-dependent DNA mechanics across high-resolution experiments and
multi-scale simulations&rdquo;</em>.</p>

<h3>Contact</h3>
<p>Please contact Rahul Sharma at
<a href="mailto:rs25.iitr@gmail.com">rs25.iitr@gmail.com</a>.</p>

<h3>Source code</h3>
<p>The source code is available on GitHub (<code>XX</code>) and can be downloaded and modified for
bespoke analysis.</p>

<div class="credit-footer">
  Created by <strong>Rahul Sharma</strong>
</div>
"""),
            ),
        ),

        # ---- Main analysis tab (left sidebar + control bar + plots) ----
        ui.nav_panel(
            "Analysis",

            ui.layout_sidebar(
                # ============================================================
                # LEFT SIDEBAR — datasets only
                # ============================================================
                ui.sidebar(
                    ui.HTML('<div class="sb-title">Datasets</div>'),
                    ui.HTML(
                        '<div class="info-box" style="margin-bottom:6px;">'
                        'Built-in datasets load automatically. Upload a CSV to '
                        '<b>replace</b> any built-in, or add a custom dataset below.'
                        '</div>'
                    ),
                    ui.input_file("file_xray",  "Replace X-ray (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.input_file("file_cryoem", "Replace Cryo-EM (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.input_file("file_md",    "Replace MD (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.input_file("file_cgdna", "Replace cgNA+ (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.div(
                        {"class": "custom-upload-block"},
                        ui.HTML('<div class="sb-title">Custom dataset (optional)</div>'),
                        ui.input_text("custom_label", "Name", value="Custom",
                                      placeholder="e.g. My simulation"),
                        ui.input_file("file_custom", "Upload CSV",
                                      accept=[".csv"], multiple=False),
                    ),
                    ui.output_ui("dataset_status_ui"),
                    width=250,
                    open="open",
                    bg="#f9fafc",
                ),

                # ============================================================
                # MAIN AREA — compact single-row control bar + tabbed plots
                # ============================================================
                ui.div(
                    {"class": "ctrl-bar"},
                    ui.layout_columns(
                        # Axes + coordinate
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Axes &amp; Coordinate</div>'),
                            ui.output_ui("dataset_x_ui"),
                            ui.output_ui("dataset_y_ui"),
                            ui.output_ui("coord_select_ui"),
                        ),
                        # Filters
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Filters</div>'),
                            ui.input_select(
                                "info", "Info",
                                choices={"Shape": "Shape", "Variance": "Variance"},
                                selected="Shape",
                            ),
                            ui.input_select(
                                "level", "Level",
                                choices={"tet": "Tetramer", "dim": "Dimer"},
                                selected="tet",
                            ),
                            ui.input_select(
                                "step_class_filter", "Step class",
                                choices={"All":"All","RR":"RR","RY":"RY","YR":"YR","YY":"YY"},
                                selected="All",
                            ),
                        ),
                        # Options
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Options</div>'),
                            ui.input_checkbox("show_identity",   "y = x line", value=True),
                            ui.input_checkbox("show_regression", "Regression", value=True),
                        ),
                        # Figure
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Figure</div>'),
                            ui.input_numeric("fig_width",    "W",          DEFAULT_FIG_W, min=400, max=2000, step=50),
                            ui.input_numeric("fig_height",   "H",          DEFAULT_FIG_H, min=300, max=1600, step=50),
                            ui.input_numeric("fig_fontsize", "Axis font",  13, min=8, max=24, step=1),
                            ui.input_numeric("fig_xfont",    "Tick/legend", 9, min=5, max=20, step=1),
                        ),
                        # Figure labels + Filename + Export (combined column)
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Figure labels (optional)</div>'),
                            ui.input_text("custom_title", None, value="",
                                          placeholder="Plot title"),
                            ui.input_text("custom_yaxis", None, value="",
                                          placeholder="Y-axis label"),
                            ui.input_text("custom_xaxis", None, value="",
                                          placeholder="X-axis label"),
                            ui.HTML('<div class="ctrl-group-label" style="margin-top:8px;">Filename &amp; Export</div>'),
                            ui.input_text("dl_filename", None, value="tetramer_plot",
                                          placeholder="filename (no ext.)"),
                            ui.div(
                                {"style": "display:flex; gap:4px; margin-top:5px; flex-wrap:wrap;"},
                                ui.download_button("dl_png", "PNG"),
                                ui.download_button("dl_pdf", "PDF"),
                                ui.download_button("dl_svg", "SVG"),
                                ui.download_button("dl_csv", "CSV"),
                            ),
                        ),
                        col_widths=[4, 2, 2, 2, 2],
                    ),
                ),


                ui.navset_tab(

                    # ---- Tab 1: Scatter ----
                    ui.nav_panel(
                        "Scatter comparison",
                        ui.output_ui("kpi_row_ui"),
                        ui.tags.br(),
                        output_widget("scatter_plot"),
                        ui.output_ui("diff_stats_ui"),
                        ui.tags.br(),
                        ui.tags.h5("Comparison table",
                                   style="margin:6px 0 4px 0; font-size:0.93rem; color:#374151;"),
                        ui.output_data_frame("comparison_table"),
                    ),

                    # ---- Tab 2: Overview ----
                    ui.nav_panel(
                        "Coordinate overview",
                        ui.HTML('<div class="info-box">Pearson r and cosine across all '
                                'coordinates for the selected dataset pair, Info (Shape / Variance) and level.</div>'),
                        ui.tags.br(),
                        output_widget("overview_bar_plot"),
                        ui.tags.br(),
                        ui.output_data_frame("overview_table"),
                    ),

                    # ---- Tab 5: Difference ----
                    ui.nav_panel(
                        "Difference plot",
                        ui.layout_columns(
                            ui.output_ui("diff_a_ui"),
                            ui.output_ui("diff_b_ui"),
                            ui.input_select(
                                "diff_sort", "Sort by",
                                choices={"alpha": "Alphabetical",
                                         "abs_diff": "Absolute difference"},
                                selected="abs_diff",
                            ),
                            ui.input_checkbox("diff_hline", "Show zero line", value=True),
                            col_widths=[3, 3, 3, 3],
                        ),
                        output_widget("difference_plot"),
                    ),

                    # ---- Tab 6: Coordinate profile ----
                    ui.nav_panel(
                        "Coordinate profile",
                        ui.layout_columns(
                            ui.input_select(
                                "profile_sort", "Sort by",
                                choices={
                                    "alpha":    "Alphabetical",
                                    "xray":     "X-ray value",
                                    "md":       "MD value",
                                    "cryoem":   "Cryo-EM value",
                                    "cgdna":    "cgNA+ value",
                                    "stepclass":"Step class",
                                },
                                selected="alpha",
                            ),
                            ui.output_ui("profile_dataset_selector_ui"),
                            ui.div(
                                ui.input_checkbox("profile_err", "Error bars  (± k·√variance)", value=False),
                                ui.input_numeric("profile_k", "k (σ multiplier)", 1, min=0.5, max=5, step=0.5),
                            ),
                            col_widths=[3, 5, 4],
                        ),
                        ui.output_ui("profile_err_note_ui"),
                        output_widget("profile_plot"),
                    ),
                    id="plot_tab",
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    # Per-session input accessors (closures over the server's `input` object)
    def _input_val(input_id: str, default=None):
        """Read an input value safely; returns default if not yet initialised."""
        try:
            return input[input_id]()
        except Exception:
            return default

    def _input_exists(input_id: str) -> bool:
        try:
            input[input_id]()
            return True
        except Exception:
            return False

    # ================================================================
    # Dataset parsing (one per upload slot)
    # ================================================================

    def _parse_slot(file_input, name: str):
        fi = file_input()
        if fi is None:
            return None, [], []
        try:
            path = fi[0]["datapath"]
            df_raw, load_err = load_dataset(path)
        except Exception as exc:
            return None, [f"{name}: unexpected error: {exc}"], []
        if load_err:
            return None, [f"{name}: {load_err}"], []
        df, warns = validate_dataset(df_raw, name)
        if df is None:
            return None, warns, []
        return df, [], warns

    @reactive.calc
    def _xray():   return _parse_slot(input.file_xray,   "X-ray (uploaded)")

    @reactive.calc
    def _cryoem(): return _parse_slot(input.file_cryoem, "Cryo-EM (uploaded)")

    @reactive.calc
    def _md():     return _parse_slot(input.file_md,     "MD (uploaded)")

    @reactive.calc
    def _cgdna():  return _parse_slot(input.file_cgdna,  "cgNA+ (uploaded)")

    @reactive.calc
    def _custom():
        lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
        return _parse_slot(input.file_custom, lbl)

    @reactive.calc
    def all_datasets():
        """
        Build the active dataset registry (rows of both levels kept together).

        Priority for the three built-in slots:
          1. User-uploaded file (overrides built-in)
          2. Built-in default from data/

        A fourth 'custom' slot is added if a file has been uploaded there.
        """
        avail = {}

        for key, builtin_label in [("xray", "X-ray"), ("cryoem", "Cryo-EM"), ("md", "MD"), ("cgdna", "cgNA+")]:
            slot_func = {"xray": _xray, "cryoem": _cryoem, "md": _md, "cgdna": _cgdna}[key]
            uploaded_df, errs, _ = slot_func()
            if uploaded_df is not None and not errs:
                # User uploaded a replacement — use it with the slot's default label
                avail[key] = (builtin_label, uploaded_df)
            else:
                # Fall back to built-in
                builtin_df, builtin_lbl = BUILTIN_DATASETS.get(key, (None, builtin_label))
                if builtin_df is not None:
                    avail[key] = (builtin_lbl, builtin_df)

        # Custom dataset
        custom_df, custom_errs, _ = _custom()
        if custom_df is not None and not custom_errs:
            lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
            avail["custom"] = (lbl, custom_df)

        return avail

    def _level() -> str:
        return "dim" if _input_val("level", "tet") == "dim" else "tet"

    def _info() -> str:
        return "Variance" if _input_val("info", "Shape") == "Variance" else "Shape"

    def _seq_word(plural: bool = False) -> str:
        w = "dimer" if _level() == "dim" else "tetramer"
        return w + ("s" if plural else "")

    def _coord_disp(coord: str) -> str:
        """Coordinate label used in titles/axes, e.g. 'Buckle (Intra), variance'."""
        lbl = f"{coord} {get_coord_type_label(coord)}".strip()
        return f"{lbl}, {_info().lower()}"

    def _counts_txt(df) -> str:
        return " · ".join(f"{i}: {c['tet']} tet / {c['dim']} dim"
                          for i, c in describe_dataset(df).items()) or "no rows"

    @reactive.calc
    def available_datasets():
        """
        Datasets that have rows for the selected Info (Shape / Variance) at the selected
        level (tetramer = 4-letter rows, dimer = 2-letter rows).  Everything downstream
        uses this view, so each plot/table automatically follows the filters.
        """
        lvl, info = _level(), _info()
        out = {}
        for key, (label, df) in all_datasets().items():
            sub = select_view(df, lvl, info)
            if sub is not None:
                out[key] = (label, sub)
        return out

    # ================================================================
    # Dynamic sidebar UI
    # ================================================================

    @output
    @render.ui
    def dataset_status_ui():
        msgs = []

        # Report upload status for the three replaceable slots
        for key, label, slot in [
            ("xray",  "X-ray", _xray),
            ("cryoem", "Cryo-EM", _cryoem),
            ("md",    "MD",    _md),
            ("cgdna", "cgNA+", _cgdna),
        ]:
            df, errs, warns = slot()
            for e in errs:
                msgs.append(ui.HTML(f'<div class="err-box">&#10006; {e}</div>'))
            for w in warns:
                msgs.append(ui.HTML(f'<div class="warn-box">&#9888; {w}</div>'))
            if df is not None and not errs:
                msgs.append(ui.HTML(
                    f'<div class="ok-box">&#10003; {label} replaced — '
                    + _counts_txt(df) + '</div>'
                ))

        # Report custom dataset
        custom_df, custom_errs, custom_warns = _custom()
        for e in custom_errs:
            msgs.append(ui.HTML(f'<div class="err-box">&#10006; {e}</div>'))
        for w in custom_warns:
            msgs.append(ui.HTML(f'<div class="warn-box">&#9888; {w}</div>'))
        if custom_df is not None and not custom_errs:
            lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
            msgs.append(ui.HTML(
                f'<div class="ok-box">&#10003; Custom "{lbl}" — '
                + _counts_txt(custom_df) + '</div>'
            ))

        # Summary of what is actually active (built-ins + overrides + custom)
        avail = all_datasets()
        active_labels = []
        for k, (lbl, df) in avail.items():
            active_labels.append(f"{lbl} ({', '.join(describe_dataset(df)) or 'no data'})")
        if avail:
            msgs.append(ui.HTML(
                f'<div class="info-box" style="margin-top:8px;">'
                f'<b>Active datasets (Info available):</b> {", ".join(active_labels)}</div>'
            ))
        else:
            msgs.append(ui.HTML(
                '<div class="warn-box">No datasets loaded. '
                'Built-in CSVs missing from <code>data/</code>.</div>'
            ))
        return ui.tags.div(*msgs)

    @output
    @render.ui
    def dataset_x_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"xray": "X-ray (not loaded)"}
        return ui.input_select("ds_x", "X-axis dataset", choices=choices,
                               selected=list(choices.keys())[0])

    @output
    @render.ui
    def dataset_y_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"md": "MD (not loaded)"}
        keys = list(choices.keys())
        default = keys[1] if len(keys) > 1 else keys[0]
        return ui.input_select("ds_y", "Y-axis dataset", choices=choices, selected=default)

    @output
    @render.ui
    def coord_select_ui():
        avail = available_datasets()
        if not avail:
            return ui.input_select("coord", "Coordinate",
                                   choices={"": "(upload data first)"}, selected="")
        df_first = list(avail.values())[0][1]
        choices = {c: f"{c} {get_coord_type_label(c)}".strip() for c in df_first.columns}
        return ui.input_select("coord", "Coordinate", choices=choices,
                               selected=list(choices.keys())[0])

    # ================================================================
    # Core reactive: get the two selected dfs and their labels
    # ================================================================

    @reactive.calc
    def selected_pair():
        """Return (label_x, df_x, label_y, df_y) or (error_str, None, None, None)."""
        avail = available_datasets()
        if len(avail) < 2:
            return "Need at least two valid datasets.", None, None, None

        keys = list(avail.keys())
        ds_x = _input_val("ds_x", keys[0])
        ds_y = _input_val("ds_y", keys[min(1, len(keys)-1)])

        # Prevent identical axes
        if ds_x == ds_y:
            ds_y = keys[(keys.index(ds_x) + 1) % len(keys)]

        if ds_x not in avail or ds_y not in avail:
            return "Selected dataset not loaded.", None, None, None

        lx, dfx = avail[ds_x]
        ly, dfy = avail[ds_y]
        return None, (lx, dfx), (ly, dfy), None

    # ================================================================
    # Filtered comparison dataframe for the active coordinate
    # ================================================================

    @reactive.calc
    def comparison_df():
        err, pair_x, pair_y, _ = selected_pair()
        if err:
            return None, err

        lx, dfx = pair_x
        ly, dfy = pair_y

        coord = _input_val("coord", None)
        if not coord:
            shared, _, _ = check_dataset_compatibility(dfx, dfy)
            if not shared:
                return None, "No shared coordinate columns."
            coord = shared[0]

        if coord not in dfx.columns or coord not in dfy.columns:
            return None, f"Coordinate '{coord}' not in both datasets."

        try:
            df = prepare_comparison(dfx, dfy, coord, lx, ly)
        except Exception as exc:
            return None, f"Comparison error: {exc}"

        cls = _input_val("step_class_filter", "All")
        if cls != "All":
            df = df[df["Step_class"] == cls].copy()


        if df.empty:
            return None, f"No {_seq_word(True)} match the current filter."

        return df, None

    # ================================================================
    # KPI row
    # ================================================================

    @output
    @render.ui
    def kpi_row_ui():
        df, err = comparison_df()
        if err:
            return ui.HTML(f'<div class="warn-box">{err}</div>')

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0]; ly = pair_y[0]
        coord = _input_val("coord", "")
        type_lbl = detect_coord_type(coord)

        x = df[lx].values; y = df[ly].values
        r, p = calculate_pearson(x, y)
        cos = calculate_cosine(x, y)

        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "—"
        p_str = f"p = {p:.2e}" if not np.isnan(p) else ""

        return ui.HTML(
            '<div class="stat-row">'
            + _kpi("Pearson r", fmt(r), p_str)
            + _kpi("Cosine similarity", fmt(cos))
            + _kpi(f"N {_seq_word(True)}", str(len(df)))
            + _kpi("Coordinate", coord, (type_lbl + " · " if type_lbl != "Unknown" else "") + _info())
            + '</div>'
        )

    # ================================================================
    # Figure helpers and the scatter plot
    # ================================================================

    @reactive.calc
    def _fig_dims():
        w   = _input_val("fig_width",    DEFAULT_FIG_W) or DEFAULT_FIG_W
        h   = _input_val("fig_height",   DEFAULT_FIG_H) or DEFAULT_FIG_H
        fs  = _input_val("fig_fontsize", 13) or 13
        xfs = _input_val("fig_xfont",   9)  or 9
        return int(w), int(h), int(fs), int(xfs)

    def _empty_fig(msg: str):
        w, h, fs, xfs = _fig_dims()
        fig = go.Figure()
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(text=msg, x=0.5),
        )
        return fig

    def _build_scatter():
        df, err = comparison_df()
        w, h, fs, xfs = _fig_dims()

        if err or df is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0]; ly = pair_y[0]
        coord    = _input_val("coord", "")
        coord_disp = _coord_disp(coord)

        show_id  = _input_val("show_identity",  True)
        show_reg = _input_val("show_regression", True)
        cls_filt = _input_val("step_class_filter", "All")

        fig = go.Figure()
        classes_to_plot = (
            [cls_filt] if cls_filt != "All"
            else [c for c in ["RR","RY","YR","YY"] if c in df["Step_class"].values]
        )

        for cls in ["RR","RY","YR","YY"]:
            if cls not in classes_to_plot:
                continue
            sub = df[df["Step_class"] == cls]
            hover = [
                f"<b>{_seq_word().capitalize()}:</b> {row['Tetramer']}<br>"
                f"<b>Central step:</b> {row['Central_step']} ({row['Step_class']})<br>"
                f"<b>{lx}:</b> {row[lx]:.4f}<br>"
                f"<b>{ly}:</b> {row[ly]:.4f}<br>"
                f"<b>Diff (Y−X):</b> {row['Difference']:.4f}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub[lx], y=sub[ly],
                mode="markers",
                name=STEP_CLASS_LABELS.get(cls, cls),
                marker=dict(color=CLASS_COLOURS[cls], size=7, opacity=0.82,
                            line=dict(width=0.5, color="white")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        # Identity line
        if show_id:
            all_v = pd.concat([df[lx], df[ly]])
            lo, hi = float(all_v.min()), float(all_v.max())
            pad = (hi - lo) * 0.05
            fig.add_shape(type="line",
                          x0=lo-pad, y0=lo-pad, x1=hi+pad, y1=hi+pad,
                          line=dict(color="#888888", dash="dash", width=1.2),
                          layer="below")

        # Regression
        reg_str = ""
        if show_reg:
            reg = linear_regression(df[lx].values, df[ly].values)
            if reg:
                fig.add_trace(go.Scatter(
                    x=reg["x_line"], y=reg["y_line"],
                    mode="lines",
                    name=f"Regression (R²={reg['r_squared']:.3f})",
                    line=dict(color="#e63946", width=1.8),
                    hoverinfo="skip",
                ))
                sgn = "+" if reg["intercept"] >= 0 else "−"
                reg_str = (
                    f"y = {reg['slope']:.3f}x {sgn} {abs(reg['intercept']):.3f}  "
                    f"R²={reg['r_squared']:.3f}"
                )

        # Annotation
        r, p = calculate_pearson(df[lx].values, df[ly].values)
        cos   = calculate_cosine(df[lx].values, df[ly].values)
        ann   = f"r = {r:.4f}  |  cos = {cos:.4f}"
        if reg_str:
            ann += f"<br>{reg_str}"
        fig.add_annotation(
            x=1.0, y=0.01, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=ann, showarrow=False,
            font=dict(size=10, color="#555555"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1, borderpad=4,
        )

        # Allow user overrides for title and axis labels
        _custom_title = (_input_val("custom_title", "") or "").strip()
        _custom_yaxis = (_input_val("custom_yaxis", "") or "").strip()
        _custom_xaxis = (_input_val("custom_xaxis", "") or "").strip()
        _plot_title  = _custom_title if _custom_title else f"{lx} vs {ly}  ·  {coord_disp}"
        _yaxis_label = _custom_yaxis if _custom_yaxis else f"{ly} — {coord_disp}"
        _xaxis_label = _custom_xaxis if _custom_xaxis else f"{lx} — {coord_disp}"

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(text=_plot_title, x=0.5, font=dict(size=fs+1)),
            xaxis_title=_xaxis_label,
            yaxis_title=_yaxis_label,
        )
        fig.update_layout(legend_font_size=xfs)
        return fig

    @render_widget
    def scatter_plot():
        return _build_scatter()

    # ================================================================
    # Difference statistics card
    # ================================================================

    @output
    @render.ui
    def diff_stats_ui():
        df, err = comparison_df()
        if err or df is None:
            return ui.HTML("")
        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            return ui.HTML("")
        lx = pair_x[0]; ly = pair_y[0]
        stats = calculate_summary_statistics(df[lx].values, df[ly].values)
        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "—"
        return ui.HTML(
            "<b style='font-size:0.82rem;color:#6b7280;'>Difference statistics (Y − X)</b>"
            "<br><br>"
            '<div class="stat-row">'
            + _kpi("Mean |diff|",   fmt(stats["mean_abs_diff"]))
            + _kpi("Median |diff|", fmt(stats["median_abs_diff"]))
            + _kpi("RMSE",          fmt(stats["rmse"]))
            + _kpi("Max |diff|",    fmt(stats["max_abs_diff"]))
            + '</div>'
        )

    # ================================================================
    # Data table
    # ================================================================

    @output
    @render.data_frame
    def comparison_table():
        df, err = comparison_df()
        if err or df is None:
            return render.DataTable(pd.DataFrame({"Message": [err or "No data"]}))
        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            return render.DataTable(pd.DataFrame())
        lx = pair_x[0]; ly = pair_y[0]
        cols = ["Tetramer","Central_step","Step_class", lx, ly, "Difference","Abs_difference"]
        cols = [c for c in cols if c in df.columns]
        tbl = df[cols].copy().rename(columns={"Tetramer": _seq_word().capitalize()})
        for c in [lx, ly, "Difference", "Abs_difference"]:
            if c in tbl.columns:
                tbl[c] = tbl[c].round(5)
        return render.DataTable(tbl, selection_mode="none", filters=True)

    # ================================================================
    # Overview stats (shared reactive)
    # ================================================================

    @reactive.calc
    def overview_stats():
        err, pair_x, pair_y, _ = selected_pair()
        if err:
            return None, err
        lx, dfx = pair_x
        ly, dfy = pair_y
        shared, _, _ = check_dataset_compatibility(dfx, dfy)
        if not shared:
            return None, "No shared coordinate columns."

        # Apply filters to shared index
        shared_idx = dfx.index.intersection(dfy.index)
        cls = _input_val("step_class_filter", "All")
        if cls != "All":
            shared_idx = shared_idx[[classify_central_step(t)[1] == cls for t in shared_idx]]
        if len(shared_idx) == 0:
            return None, f"No {_seq_word(True)} match the filter."

        stats = calculate_all_coords_stats(dfx.loc[shared_idx], dfy.loc[shared_idx], shared)
        return stats, None

    # ================================================================
    # Tab 2: Coordinate overview
    # ================================================================

    @render_widget
    def overview_bar_plot():
        return _build_overview()

    def _build_overview():
        stats, err = overview_stats()
        w, h, fs, xfs = _fig_dims()
        if err or stats is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0] if pair_x else "X"
        ly = pair_y[0] if pair_y else "Y"

        hover = [
            f"<b>{row['Coordinate']}</b><br>Pearson r: {row['Pearson_r']:.4f}<br>"
            f"Cosine: {row['Cosine']:.4f}<br>N: {row['N']}"
            for _, row in stats.iterrows()
        ]
        bar_colors = [
            "#3b5bdb" if detect_coord_type(c) == "Intra"
            else "#e63946" if detect_coord_type(c) == "Inter"
            else "#555555"
            for c in stats["Coordinate"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats["Coordinate"], y=stats["Pearson_r"],
            marker_color=bar_colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#999999", line_width=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color="#cccccc", line_width=1)
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=int(h * 0.85),
            title=dict(text=f"Pearson r — {lx} vs {ly}  ·  {_info().lower()}, {_seq_word()} level", x=0.5, font=dict(size=fs+1)),
            xaxis_title="Coordinate",
            yaxis_title="Pearson r",
            showlegend=False,
        )
        # Extend the y-axis range without spreading a duplicate yaxis key
        fig.update_yaxes(range=[-0.1, 1.05])
        fig.update_xaxes(tickfont=dict(size=xfs))
        return fig

    @output
    @render.data_frame
    def overview_table():
        stats, err = overview_stats()
        if err or stats is None:
            return render.DataTable(pd.DataFrame({"Message": [err or "No data"]}))
        tbl = stats.copy()
        tbl["Pearson_r"] = tbl["Pearson_r"].round(4)
        tbl["Cosine"]    = tbl["Cosine"].round(4)
        return render.DataTable(tbl, selection_mode="none", filters=False)

    # ================================================================
    # Tab 5: Difference bar plot
    # ================================================================

    @output
    @render.ui
    def diff_a_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"xray": "(no data)"}
        keys = list(choices)
        with reactive.isolate():
            prev = _input_val("diff_a", None)
        sel = prev if prev in choices else keys[min(1, len(keys) - 1)]
        return ui.input_select("diff_a", "Dataset A (A − B)", choices=choices, selected=sel)

    @output
    @render.ui
    def diff_b_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"xray": "(no data)"}
        keys = list(choices)
        with reactive.isolate():
            prev = _input_val("diff_b", None)
        sel = prev if prev in choices else keys[0]
        return ui.input_select("diff_b", "Dataset B (subtracted)", choices=choices, selected=sel)

    @reactive.calc
    def diff_df():
        """Difference table A - B for the chosen coordinate, honouring the step filters."""
        avail = available_datasets()
        if len(avail) < 2:
            return None, None, None, "Need at least two valid datasets."
        keys = list(avail)
        ka = _input_val("diff_a", keys[min(1, len(keys) - 1)])
        kb = _input_val("diff_b", keys[0])
        if ka not in avail or kb not in avail:
            return None, None, None, "Selected dataset not loaded."
        if ka == kb:
            return None, None, None, "Pick two different datasets for A and B."
        la, dfa = avail[ka]
        lb, dfb = avail[kb]
        if la == lb:                         # guard against identical display names
            lb = lb + " (B)"
        coord = _input_val("coord", None)
        if not coord:
            shared, _, _ = check_dataset_compatibility(dfa, dfb)
            if not shared:
                return None, None, None, "No shared coordinate columns."
            coord = shared[0]
        if coord not in dfa.columns or coord not in dfb.columns:
            return None, None, None, f"Coordinate '{coord}' not in both datasets."
        # prepare_comparison returns (second - first); put B first so Difference = A - B
        df = prepare_comparison(dfb, dfa, coord, lb, la)
        cls = _input_val("step_class_filter", "All")
        if cls != "All":
            df = df[df["Step_class"] == cls].copy()
        if df.empty:
            return None, None, None, f"No {_seq_word(True)} match the current filter."
        return df, la, lb, None

    @render_widget
    def difference_plot():
        return _build_difference()

    def _build_difference():
        df, la, lb, err = diff_df()
        w, h, fs, xfs = _fig_dims()
        if err or df is None:
            return _empty_fig(err or "No data")

        coord    = _input_val("coord", "")
        sort_by  = _input_val("diff_sort",  "abs_diff")
        show_zl  = _input_val("diff_hline", True)

        data = df.copy()
        if sort_by == "abs_diff":
            data = data.sort_values("Abs_difference", ascending=False)
        else:
            data = data.sort_values("Tetramer")

        fig = go.Figure()
        cls_filt = _input_val("step_class_filter", "All")
        classes  = [cls_filt] if cls_filt != "All" else ["RR","RY","YR","YY"]

        for cls in classes:
            sub = data[data["Step_class"] == cls]
            if sub.empty:
                continue
            hover = [
                f"<b>{row['Tetramer']}</b> ({row['Central_step']}, {row['Step_class']})<br>"
                f"{la} − {lb}: {row['Difference']:.4f}<br>"
                f"{la}: {row[la]:.4f}   {lb}: {row[lb]:.4f}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Bar(
                x=sub["Tetramer"], y=sub["Difference"],
                name=STEP_CLASS_LABELS.get(cls, cls),
                marker_color=CLASS_COLOURS[cls],
                opacity=0.85,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        if show_zl:
            fig.add_hline(y=0, line_dash="dot", line_color="#777777", line_width=1.2)

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(
                text=f"Difference ({la} − {lb})  ·  {_coord_disp(coord)}",
                x=0.5, font=dict(size=fs+1),
            ),
            xaxis_title=_seq_word().capitalize(),
            yaxis_title=f"Difference in {_info().lower()} ({la} − {lb})",
            barmode="relative",
        )
        fig.update_layout(legend_font_size=xfs)
        n_bars = len(data)
        auto_tick_fs = max(5, min(11, int(round(400 / max(n_bars, 1)))))
        tick_fs = max(auto_tick_fs, xfs)
        fig.update_xaxes(tickangle=90, tickfont=dict(size=tick_fs))
        return fig

    # ================================================================
    # Coordinate profile (with optional +/- k*sqrt(variance) error bars)
    # ================================================================

    @output
    @render.ui
    def profile_dataset_selector_ui():
        # Always offer every dataset slot; slots without data are simply skipped in the plot.
        custom_lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
        choices = {
            "xray":   "X-ray",
            "cryoem": "Cryo-EM",
            "md":     "MD",
            "cgdna":  "cgNA+",
            "custom": custom_lbl,
        }
        # Keep the user's current ticks across re-renders; default = everything ticked
        with reactive.isolate():
            prev = _input_val("profile_datasets", None)
        selected = [k for k in (prev or choices.keys()) if k in choices]
        return ui.input_checkbox_group(
            "profile_datasets",
            "Datasets to show",
            choices=choices,
            selected=selected,
            inline=True,
        )

    def _rgba(hex_color: str, alpha: float) -> str:
        h = str(hex_color).lstrip("#")
        if len(h) == 6:
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        return hex_color

    @reactive.calc
    def profile_data():
        """Table behind the Coordinate-profile plot (also used for the CSV export).

        Returns (dict, None) or (None, message).  With error bars enabled (Info = Shape),
        `merged` also holds one `<key>__sd` column = k * sqrt(variance) for every dataset
        that has Variance rows at the selected level.
        """
        avail = available_datasets()
        coord = _input_val("coord", None)
        if not avail:
            return None, "Upload at least one dataset."
        if not coord:
            return None, "Select a coordinate in the control bar."

        selected_ds = _input_val("profile_datasets", list(avail.keys()))
        if not selected_ds:
            return None, "Select at least one dataset to display."
        if isinstance(selected_ds, str):
            selected_ds = [selected_ds]

        frames = {}
        for key in selected_ds:
            if key not in avail:
                continue
            lbl, df = avail[key]
            if coord not in df.columns:
                continue
            frames[key] = df[coord].rename(key)
        if not frames:
            return None, f"Coordinate '{coord}' not found in any selected dataset."

        merged = pd.concat(frames.values(), axis=1).dropna(how="all")
        merged.index.name = "Tetramer"
        merged = merged.reset_index()
        merged["Central_step"] = merged["Tetramer"].apply(lambda t: classify_central_step(t)[0])
        merged["Step_class"] = merged["Tetramer"].apply(lambda t: classify_central_step(t)[1])

        cls_filt = _input_val("step_class_filter", "All")
        if cls_filt != "All":
            merged = merged[merged["Step_class"] == cls_filt].copy()
        if merged.empty:
            return None, f"No {_seq_word(True)} match the current filter."

        sort_by = _input_val("profile_sort", "alpha")
        if sort_by == "stepclass":
            merged = merged.sort_values(["Step_class", "Central_step", "Tetramer"])
        elif sort_by in frames:
            merged = merged.sort_values(sort_by, ascending=True)
        else:
            merged = merged.sort_values("Tetramer")

        # ---- error bars: k * sqrt(variance) of the same dataset / level / coordinate ----
        want_err = bool(_input_val("profile_err", False))
        try:
            k = float(_input_val("profile_k", 1) or 1)
        except (TypeError, ValueError):
            k = 1.0
        k = max(k, 0.0)
        err_on = want_err and _info() == "Shape"
        have_sd, missing = [], []
        if err_on:
            full = all_datasets()
            for key in frames:
                var_df = select_view(full[key][1], _level(), "Variance") if key in full else None
                if var_df is None or coord not in var_df.columns:
                    missing.append(avail[key][0])
                    continue
                sd = np.sqrt(var_df[coord].clip(lower=0))
                merged[f"{key}__sd"] = k * sd.reindex(merged["Tetramer"]).values
                have_sd.append(key)

        return dict(merged=merged, keys=[k_ for k_ in selected_ds if k_ in frames], avail=avail,
                    coord=coord, k=k, want_err=want_err, err_on=err_on,
                    have_sd=have_sd, missing=missing), None

    @output
    @render.ui
    def profile_err_note_ui():
        if not _input_val("profile_err", False):
            return ui.HTML("")
        if _info() != "Shape":
            return ui.HTML('<div class="warn-box">Error bars (&plusmn;k&middot;&radic;variance) are drawn '
                           'around the <b>Shape</b> values. Set Info to Shape to use them.</div>')
        d, err = profile_data()
        if d is None:
            return ui.HTML("")
        msg = (f"Error bars: &plusmn;{d['k']:g} &times; &radic;variance "
               f"({_seq_word()} level, same dataset).")
        box = "info-box"
        if d["missing"]:
            msg += " No Variance rows for: " + ", ".join(d["missing"]) + " (no error bars drawn)."
            box = "warn-box"
        return ui.HTML(f'<div class="{box}">{msg}</div>')

    def _build_profile():
        d, err = profile_data()
        w, h, fs, xfs = _fig_dims()
        if d is None:
            return _empty_fig(err or "No data")
        merged, avail, coord, k = d["merged"], d["avail"], d["coord"], d["k"]
        coord_disp = _coord_disp(coord)

        fig = go.Figure()
        for key in d["keys"]:
            mk = DATASET_MARKERS.get(key, dict(symbol="circle", color="#555555", size=6))
            label = avail[key][0]
            has_sd = key in d["have_sd"]
            sd_col = merged[f"{key}__sd"] if has_sd else None
            hover = []
            for i, (_, row) in enumerate(merged.iterrows()):
                txt = (f"<b>{row['Tetramer']}</b><br>"
                       f"Central step: {row['Central_step']} ({row['Step_class']})<br>"
                       f"{label}: {row[key]:.4f}")
                if has_sd and pd.notna(sd_col.iloc[i]):
                    txt += f" &plusmn; {sd_col.iloc[i]:.4f}"
                hover.append(txt)
            trace = dict(
                x=merged["Tetramer"], y=merged[key], mode="markers", name=label,
                marker=dict(symbol=mk["symbol"], color=mk["color"], size=mk["size"],
                            opacity=0.80, line=dict(width=0.6, color="white")),
                hovertemplate="%{customdata}<extra></extra>", customdata=hover,
            )
            if has_sd:
                trace["error_y"] = dict(type="data", array=sd_col.fillna(0).values, visible=True,
                                        thickness=1, width=2, color=_rgba(mk["color"], 0.55))
            fig.add_trace(go.Scatter(**trace))

        title = f"Coordinate profile — {coord_disp}"
        if d["have_sd"]:
            title += f"  (error bars: ±{k:g}·√variance)"
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(text=title, x=0.5, font=dict(size=fs + 1)),
            xaxis_title=_seq_word().capitalize(),
            yaxis_title=coord_disp,
        )
        fig.update_layout(legend_font_size=xfs)
        n_pts = len(merged)
        auto_tick_fs = max(5, min(11, int(round(400 / max(n_pts, 1)))))
        tick_fs = max(auto_tick_fs, xfs)
        fig.update_xaxes(tickangle=90, tickfont=dict(size=tick_fs))
        return fig

    @render_widget
    def profile_plot():
        return _build_profile()

    # ================================================================
    # Downloads - always export what the ACTIVE tab shows
    # ================================================================

    def _active_tab() -> str:
        return _input_val("plot_tab", "Scatter comparison") or "Scatter comparison"

    def _active_fig():
        builder = {
            "Coordinate overview": _build_overview,
            "Difference plot": _build_difference,
            "Coordinate profile": _build_profile,
        }.get(_active_tab(), _build_scatter)
        return builder()

    def _export_bytes(fmt: str) -> bytes:
        """Real PNG/PDF/SVG bytes, or an error - never a placeholder under an image name."""
        try:
            return figure_to_bytes(_active_fig(), fmt)
        except ExportError as exc:
            try:
                ui.notification_show(f"Export failed: {exc}", type="error", duration=15)
            except Exception:
                pass
            raise

    def _active_table():
        """(DataFrame, error) for the CSV export of the active tab."""
        tab = _active_tab()
        word = _seq_word().capitalize()
        if tab == "Coordinate overview":
            stats, err = overview_stats()
            return stats, err
        if tab == "Difference plot":
            df, la, lb, err = diff_df()
            if df is None:
                return None, err
            cols = ["Tetramer", "Central_step", "Step_class", lb, la, "Difference", "Abs_difference"]
            return df[[c for c in cols if c in df.columns]].rename(columns={"Tetramer": word}), None
        if tab == "Coordinate profile":
            d, err = profile_data()
            if d is None:
                return None, err
            m = d["merged"].copy()
            ren = {"Tetramer": word}
            for key in d["keys"]:
                ren[key] = d["avail"][key][0]
                if f"{key}__sd" in m.columns:
                    ren[f"{key}__sd"] = f"{d['avail'][key][0]} (±{d['k']:g}·sqrt variance)"
            cols = ["Tetramer", "Central_step", "Step_class"]
            for key in d["keys"]:
                cols.append(key)
                if f"{key}__sd" in m.columns:
                    cols.append(f"{key}__sd")
            return m[cols].rename(columns=ren), None
        df, err = comparison_df()
        if df is None:
            return None, err
        _, pair_x, pair_y, _ = selected_pair()
        lx, ly = pair_x[0], pair_y[0]
        cols = ["Tetramer", "Central_step", "Step_class", lx, ly, "Difference", "Abs_difference"]
        return df[[c for c in cols if c in df.columns]].rename(columns={"Tetramer": word}), None

    def _dl_fname(ext: str) -> str:
        base = (_input_val("dl_filename", "tetramer_plot") or "tetramer_plot").strip()
        import re as _re
        base = _re.sub(r'[^\w\-\.]', '_', base).strip('_') or "tetramer_plot"
        return f"{base}.{ext}"

    @render.download_button(filename=lambda: _dl_fname("png"), media_type="image/png")
    def dl_png():
        yield _export_bytes("png")

    @render.download_button(filename=lambda: _dl_fname("pdf"), media_type="application/pdf")
    def dl_pdf():
        yield _export_bytes("pdf")

    @render.download_button(filename=lambda: _dl_fname("svg"), media_type="image/svg+xml")
    def dl_svg():
        yield _export_bytes("svg")

    @render.download_button(filename=lambda: _dl_fname("csv"), media_type="text/csv")
    def dl_csv():
        tbl, err = _active_table()
        if tbl is None:
            yield f"message\n{err or 'No data'}\n"
            return
        yield tbl.to_csv(index=False)

# ---------------------------------------------------------------------------
# Input helpers  (NOTE: these are module-level placeholders;
# the real per-session versions are created inside server() and shadow these)
# ---------------------------------------------------------------------------

def _input_exists(input_id: str) -> bool:
    return False

def _input_val(input_id: str, default=None):
    return default


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = App(app_ui, server, static_assets=_DATA_DIR)   # serves data/*.csv for the About-page example links
