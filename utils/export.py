"""
utils/export.py
===============
Static image export (PNG / PDF / SVG) for the Plotly figures shown in the app.

Strategy
--------
1. Kaleido (via ``fig.to_image``) - renders the figure exactly as displayed.
   Kaleido 0.2.x bundles its own browser engine; Kaleido 1.x needs a separate Google Chrome.
2. If Kaleido is unusable (for example Kaleido 1.x without Chrome), fall back to a
   Chrome-free matplotlib re-drawing of the same figure.  It supports the trace types used
   by this app (markers, lines, bars, error bars, reference lines, annotations).
3. If both fail, raise ``ExportError`` - the caller must NOT write a placeholder file under
   an image extension, because that produces an unreadable download.

All functions are pure (no Shiny dependency) so they can be tested on their own.
"""

from __future__ import annotations

import io
import re
import sys
import time
from typing import Optional

__all__ = ["ExportError", "figure_to_bytes", "export_engine_status", "MIME"]

MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}

# Magic numbers used to verify that what we hand to the browser really is that format.
_MAGIC = {"png": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-", "svg": b"<"}

_KALEIDO_RETRY_AFTER_S = 120.0
_kaleido_state = {"ok": None, "error": None, "t": 0.0}


class ExportError(RuntimeError):
    """Raised when a figure cannot be exported by any available engine."""


def _log(msg: str) -> None:
    print(f"[export] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Engine 1: Kaleido
# ---------------------------------------------------------------------------

def _kaleido_bytes(fig, fmt: str) -> bytes:
    scale = 2 if fmt == "png" else 1
    return fig.to_image(format=fmt, scale=scale)


def _kaleido_allowed() -> bool:
    """After a failure, do not retry on every click (each attempt can take seconds)."""
    if _kaleido_state["ok"] is False:
        return (time.time() - _kaleido_state["t"]) > _KALEIDO_RETRY_AFTER_S
    return True


# ---------------------------------------------------------------------------
# Engine 2: matplotlib fallback
# ---------------------------------------------------------------------------

_MARKERS = {
    "circle": "o", "square": "s", "diamond": "D", "hexagon": "h", "triangle-up": "^",
    "triangle-down": "v", "cross": "P", "x": "X", "star": "*", "pentagon": "p",
}
_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}
_PX_TO_PT = 0.72          # figure is built at 100 dpi, so 1 px = 0.72 pt


def _strip_html(text) -> str:
    if text is None:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", str(text), flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&nbsp;", " ").replace("&amp;", "&")


def _color(c, default="#555555"):
    """Convert a Plotly colour (hex / named / 'rgba(r,g,b,a)') to something matplotlib accepts."""
    if c is None:
        return default
    if isinstance(c, str):
        m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)", c)
        if m:
            r, g, b = (float(m.group(i)) / 255.0 for i in (1, 2, 3))
            a = float(m.group(4)) if m.group(4) is not None else 1.0
            return (r, g, b, a)
        return c
    return c


def _seq(v):
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return [v]


def _matplotlib_bytes(fig, fmt: str) -> bytes:
    import numpy as np
    import matplotlib
    from matplotlib.figure import Figure

    lay = fig.layout
    w = float(lay.width or 760)
    h = float(lay.height or 540)
    base_fs = float((lay.font.size if lay.font and lay.font.size else 13)) * _PX_TO_PT

    def fsz(obj, default=None):
        try:
            if obj is not None and obj.size:
                return float(obj.size) * _PX_TO_PT
        except Exception:
            pass
        return default if default is not None else base_fs

    import logging
    from matplotlib import font_manager
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans") if f in installed),
                  "DejaVu Sans")
    rc = {
        "font.family": [family],
        "pdf.fonttype": 42,          # keep text as real text in PDF
        "svg.fonttype": "none",      # keep text as real text in SVG
    }
    with matplotlib.rc_context(rc):
        mfig = Figure(figsize=(w / 100.0, h / 100.0), dpi=100 if fmt != "png" else 200,
                      facecolor="white")
        ax = mfig.add_subplot(111)
        ax.set_facecolor("white")

        # ---- categorical x handling: mimic Plotly's "order of first appearance" ----
        traces = [t for t in fig.data if getattr(t, "x", None) is not None]
        cats: list = []
        categorical = False
        for t in traces:
            xs = _seq(t.x)
            if xs and isinstance(xs[0], str):
                categorical = True
            if categorical:
                for v in xs:
                    if v not in cats:
                        cats.append(v)
        pos = {c: i for i, c in enumerate(cats)}

        def xpos(xs):
            xs = _seq(xs)
            return np.array([pos[v] for v in xs], dtype=float) if categorical else np.asarray(xs, dtype=float)

        handles = []
        for t in traces:
            ttype = t.type
            name = t.name
            x = xpos(t.x)
            y = np.asarray(_seq(t.y), dtype=float)
            show_leg = (t.showlegend is not False) and bool(name)
            label = name if show_leg else None

            if ttype == "bar":
                col = t.marker.color
                col = [_color(c) for c in col] if (col is not None and not isinstance(col, str)) else _color(col)
                ax.bar(x, y, width=0.8, color=col, alpha=float(t.opacity or 1.0), label=label, zorder=2)
            elif ttype == "scatter":
                mode = t.mode or "markers"
                if "lines" in mode:
                    lc = _color(t.line.color)
                    lw = float(t.line.width or 1.5) * _PX_TO_PT
                    ax.plot(x, y, color=lc, lw=lw, ls=_DASH.get(t.line.dash or "solid", "-"),
                            label=label, zorder=3)
                if "markers" in mode:
                    mk = t.marker
                    size = mk.size if isinstance(mk.size, (int, float)) else 7
                    ax.scatter(x, y, s=(float(size) * _PX_TO_PT) ** 2,
                               marker=_MARKERS.get(mk.symbol or "circle", "o"),
                               c=[_color(mk.color)] if isinstance(mk.color, str) or mk.color is None
                               else [_color(c) for c in mk.color],
                               alpha=float(mk.opacity if mk.opacity is not None else 1.0),
                               edgecolors="white", linewidths=0.5, label=label, zorder=3)
                # error bars
                ey = getattr(t, "error_y", None)
                if ey is not None and ey.array is not None and ey.visible is not False:
                    err = np.asarray(_seq(ey.array), dtype=float)
                    ok = np.isfinite(err) & np.isfinite(y)
                    if ok.any():
                        ax.errorbar(x[ok], y[ok], yerr=err[ok], fmt="none",
                                    ecolor=_color(ey.color or (t.marker.color if t.marker else None)),
                                    elinewidth=float(ey.thickness or 1) * _PX_TO_PT,
                                    capsize=float(ey.width if ey.width is not None else 2) * _PX_TO_PT,
                                    zorder=2)
            if show_leg:
                handles.append(name)

        # ---- reference lines (identity line, zero line) ----
        for s in (lay.shapes or []):
            if s.type != "line":
                continue
            ls = _DASH.get(getattr(s.line, "dash", None) or "solid", "-")
            lc = _color(getattr(s.line, "color", None), "#777777")
            lw = float(getattr(s.line, "width", None) or 1.0) * _PX_TO_PT
            if s.xref and "domain" in str(s.xref):
                ax.axhline(s.y0, color=lc, ls=ls, lw=lw, zorder=1)
            else:
                ax.plot([s.x0, s.x1], [s.y0, s.y1], color=lc, ls=ls, lw=lw, zorder=1)

        # ---- axes cosmetics ----
        ax.grid(True, color="#e8e8e8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#aaaaaa")

        xt = lay.xaxis
        tick_fs = fsz(xt.tickfont, base_fs)
        if categorical:
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=float(xt.tickangle or 0), fontsize=tick_fs,
                               ha="center" if not xt.tickangle else "right" if xt.tickangle not in (90, -90) else "center")
            ax.set_xlim(-0.8, len(cats) - 0.2)
        else:
            ax.tick_params(axis="x", labelsize=tick_fs)
        ax.tick_params(axis="y", labelsize=fsz(lay.yaxis.tickfont, base_fs))

        if lay.xaxis.title and lay.xaxis.title.text:
            ax.set_xlabel(_strip_html(lay.xaxis.title.text), fontsize=base_fs)
        if lay.yaxis.title and lay.yaxis.title.text:
            ax.set_ylabel(_strip_html(lay.yaxis.title.text), fontsize=base_fs)
        if lay.yaxis.range:
            ax.set_ylim(*lay.yaxis.range)
        if lay.xaxis.range and not categorical:
            ax.set_xlim(*lay.xaxis.range)
        if lay.title and lay.title.text:
            _pad = 30 if (lay.legend and lay.legend.orientation == "h") else 6   # leave room for a legend above the plot
            ax.set_title(_strip_html(lay.title.text), fontsize=fsz(lay.title.font, base_fs + 1), pad=_pad)

        # ---- annotations placed in paper coordinates (stats box, etc.) ----
        for a in (lay.annotations or []):
            if str(a.xref) != "paper" or not a.text:
                continue
            ax.text(float(a.x), float(a.y), _strip_html(a.text), transform=ax.transAxes,
                    ha={"left": "left", "right": "right"}.get(a.xanchor, "center"),
                    va={"top": "top", "bottom": "bottom"}.get(a.yanchor, "center"),
                    fontsize=fsz(a.font, base_fs * 0.8), color=_color(getattr(a.font, "color", None), "#555555"),
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9), zorder=5)

        # ---- legend: follow the Plotly placement (inside a corner / above / outside right) ----
        if lay.showlegend is not False and handles:
            L = lay.legend
            lfs = fsz(L.font, base_fs * 0.85)
            lx = float(L.x) if L.x is not None else 1.02
            ly = float(L.y) if L.y is not None else 1.0
            if L.orientation == "h":
                lg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                               frameon=True, fontsize=lfs, borderaxespad=0.3)
            elif 0.0 <= lx <= 1.0 and 0.0 <= ly <= 1.0:
                vert = {"top": "upper", "bottom": "lower"}.get(L.yanchor, "center")
                horiz = {"left": "left", "right": "right"}.get(L.xanchor, "center")
                loc = "center" if (vert, horiz) == ("center", "center") else f"{vert} {horiz}".replace("center left", "center left")
                lg = ax.legend(loc=loc, bbox_to_anchor=(lx, ly), frameon=True, fontsize=lfs, borderaxespad=0.0)
            else:
                lg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=lfs)
            lg.get_frame().set_edgecolor("#cccccc")
            lg.get_frame().set_alpha(0.9)
            lg.set_zorder(10)

        mfig.set_layout_engine("constrained")
        buf = io.BytesIO()
        mfig.savefig(buf, format=fmt, facecolor="white")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _looks_valid(data: Optional[bytes], fmt: str) -> bool:
    if not data or len(data) < 200:
        return False
    return data[: len(_MAGIC[fmt])] == _MAGIC[fmt] if fmt != "svg" else (b"<svg" in data[:600])


def figure_to_bytes(fig, fmt: str) -> bytes:
    """Return PNG / PDF / SVG bytes for a Plotly figure, or raise ExportError."""
    fmt = fmt.lower()
    if fmt not in MIME:
        raise ExportError(f"Unsupported export format: {fmt}")

    problems = []

    if _kaleido_allowed():
        try:
            data = _kaleido_bytes(fig, fmt)
            if _looks_valid(data, fmt):
                _kaleido_state.update(ok=True, error=None)
                return data
            problems.append("Kaleido returned data that is not a valid " + fmt.upper())
        except Exception as exc:                       # Chrome missing, wrong plotly/kaleido pair, ...
            msg = " ".join(str(exc).split())[:240]
            problems.append(f"Kaleido: {msg}")
            if _kaleido_state["ok"] is not False:
                _log(f"Kaleido unavailable ({msg}). Using the matplotlib fallback. "
                     "For pixel-exact exports install: pip install 'kaleido==0.2.1' 'plotly<7'")
            _kaleido_state.update(ok=False, error=msg, t=time.time())

    try:
        data = _matplotlib_bytes(fig, fmt)
        if _looks_valid(data, fmt):
            return data
        problems.append("matplotlib fallback produced invalid data")
    except Exception as exc:
        problems.append(f"matplotlib fallback: {' '.join(str(exc).split())[:200]}")

    raise ExportError("Could not create the " + fmt.upper() + " image. " + " | ".join(problems))


def export_engine_status() -> str:
    """Human-readable one-liner about which engine will be used (for logs / self-test)."""
    try:
        import plotly.graph_objects as go
        data = _kaleido_bytes(go.Figure(go.Scatter(x=[1, 2], y=[1, 2])), "png")
        if _looks_valid(data, "png"):
            return "Kaleido OK (pixel-exact export)"
    except Exception as exc:
        return "Kaleido NOT usable -> matplotlib fallback will be used (" + " ".join(str(exc).split())[:90] + ")"
    return "Kaleido returned invalid data -> matplotlib fallback will be used"
