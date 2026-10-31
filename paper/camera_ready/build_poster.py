#!/usr/bin/env python3
"""Build the print-ready 24 x 36 inch GW-CCA poster for LoG 2026.

Figure 1 is embedded as the paper's original vector panels
(figures/CCA.pdf, figures/DWA_CCA.pdf); equations are typeset with
matplotlib mathtext (Computer Modern) and embedded as vector forms.
Every number comes from Tables 1-3 of the camera-ready paper (main.tex).

    uv run --with reportlab --with pdfrw --with matplotlib -- python build_poster.py
"""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "38-Deng-poster.pdf"
FIG_STANDARD = ROOT / "figures" / "CCA.pdf"
FIG_GW = ROOT / "figures" / "DWA_CCA.pdf"
CODE_URL = "https://github.com/jianjinuta/gw-cca"

# 24 x 36 inch portrait (LoG maximum). 1 PDF point = 1 printed point.
PAGE_W, PAGE_H = 24 * inch, 36 * inch
MARGIN = 0.6 * inch
CONTENT_W = PAGE_W - 2 * MARGIN
GAP = 0.32 * inch

INK = HexColor("#13283A")
MUTED = HexColor("#4B5E6B")
PAPER = HexColor("#F6FAFB")
NAVY = HexColor("#0B2D3F")
NAVY_TEXT = HexColor("#D7E8EC")
TEAL = HexColor("#007C83")
TEAL_LIGHT = HexColor("#E3F3F2")
GOLD = HexColor("#C98A22")
GOLD_LIGHT = HexColor("#FFF4E0")
LINE = HexColor("#B9CED4")
SOFT = HexColor("#EEF4F5")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=25, leading=32, textColor=INK, alignment=TA_LEFT)
BODY_SMALL = ParagraphStyle("small", parent=BODY, fontSize=23, leading=29)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=19, leading=24, textColor=MUTED)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=25, leading=31, textColor=MUTED, alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=26, leading=30, alignment=TA_CENTER)

DATASETS = ["ogbn-arxiv", "Reddit", "Yelp", "ogbn-products", "PPI"]

_MATH_DIR = Path(tempfile.mkdtemp(prefix="gwcca-poster-math-"))
_math_count = 0


def paragraph(c, text, x, top, width, style):
    """Draw a paragraph hanging from `top`; return its bottom edge."""
    para = Paragraph(text, style)
    _, height = para.wrap(width, PAGE_H)
    para.drawOn(c, x, top - height)
    return top - height


def box(c, x, y, width, height, fill, stroke=None, radius=12, line_width=1.5):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.setLineWidth(line_width)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def heading(c, label, x, top, width):
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(x, top - 34, label)
    c.setStrokeColor(TEAL)
    c.setLineWidth(3)
    c.line(x, top - 50, x + width, top - 50)
    return top - 50 - 0.28 * inch


def pdf_form(path):
    page = pagexobj(PdfReader(str(path)).pages[0])
    x0, y0, x1, y1 = (float(v) for v in page.BBox)
    return page, x0, y0, x1 - x0, y1 - y0


def draw_pdf(c, path, cx, y, width, height):
    """Draw page 1 of a PDF as a vector form, fit in (width, height), centred on cx."""
    page, x0, y0, w, h = pdf_form(path)
    scale = min(width / w, height / h)
    c.saveState()
    c.translate(cx - w * scale / 2, y + (height - h * scale) / 2)
    c.scale(scale, scale)
    c.translate(-x0, -y0)
    c.doForm(makerl(c, page))
    c.restoreState()
    return w * scale, h * scale


def math(tex, size, color="#13283A"):
    """Typeset `tex` with mathtext at `size` pt; return the PDF path."""
    global _math_count
    _math_count += 1
    path = _MATH_DIR / f"eq{_math_count}.pdf"
    fig = plt.figure(figsize=(1, 1))
    fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return path


def draw_math(c, tex, cx, top, size, color="#13283A"):
    """Draw an equation at natural size, hanging from `top` and centred on cx; return its bottom."""
    path = math(tex, size, color)
    _, _, _, w, h = pdf_form(path)
    draw_pdf(c, path, cx, top - h, w, h)
    return top - h


def draw_header(c):
    header_h = 4.0 * inch
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - header_h, PAGE_W, header_h, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, PAGE_H - header_h, PAGE_W, 0.14 * inch, fill=1, stroke=0)

    title = "Graph-Weighted Canonical Correlation Analysis"
    size = min(80, 0.97 * CONTENT_W / stringWidth(title, "Helvetica-Bold", 1))
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", size)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.22 * inch, title)
    c.setFillColor(NAVY_TEXT)
    c.setFont("Helvetica", 42)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.88 * inch, "A Closed-Form Method for Inductive Attributed Graph Embedding")
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 2.56 * inch, "Jianjin Deng¹   ·   Di Ming²")
    c.setFillColor(NAVY_TEXT)
    c.setFont("Helvetica", 27)
    c.drawCentredString(
        PAGE_W / 2, PAGE_H - 3.03 * inch, "¹Independent Researcher      ²Chongqing University of Technology"
    )
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 25)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 3.55 * inch, "LoG 2026  ·  PMLR Volume 338  ·  github.com/jianjinuta/gw-cca")
    return PAGE_H - header_h


def draw_figure1(c, top):
    height = 8.75 * inch
    y = top - height
    box(c, MARGIN, y, CONTENT_W, height, white, LINE, radius=16)
    pad = 0.4 * inch
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(MARGIN + pad, top - 0.72 * inch, "CORE IDEA  ·  FIGURE 1")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 27)
    c.drawRightString(
        PAGE_W - MARGIN - pad, top - 0.72 * inch, "Replace same-index alignment with graph-weighted alignment"
    )

    half = CONTENT_W / 2
    panel_h = 5.3 * inch
    panel_y = top - 1.05 * inch - panel_h
    left_cx, right_cx = MARGIN + half / 2, MARGIN + 1.5 * half
    draw_pdf(c, FIG_STANDARD, left_cx, panel_y, half - 0.6 * inch, panel_h)
    draw_pdf(c, FIG_GW, right_cx, panel_y, half - 0.6 * inch, panel_h)
    c.setStrokeColor(LINE)
    c.setLineWidth(1.5)
    c.line(MARGIN + half, y + 0.4 * inch, MARGIN + half, top - 1.05 * inch)

    captions = [
        (
            left_cx,
            "(a) Standard CCA",
            "aligns only same-index pairs (<i>i</i>, <i>i</i>)",
            r"\max\ \mathrm{Tr}(W^{\top}\! X_1^{\top} X_2 H)",
        ),
        (
            right_cx,
            "(b) GW-CCA",
            "aligns every pair (<i>i</i>, <i>j</i>) in proportion to graph similarity <i>M<sub>ij</sub></i>",
            r"\max\ \mathrm{Tr}(\widetilde{W}^{\top}\! X_1^{\top} M X_2 \widetilde{H})",
        ),
    ]
    width = half - 0.8 * inch
    for cx, title, detail, tex in captions:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 31)
        c.drawCentredString(cx, panel_y - 0.48 * inch, title)
        bottom = paragraph(c, detail, cx - width / 2, panel_y - 0.62 * inch, width, CAPTION)
        draw_math(c, tex, cx, bottom - 0.1 * inch, 34)
    return y


def draw_problem_card(c, x, top, width):
    top = heading(c, "Problem", x, top, width)
    top = paragraph(
        c,
        "<b>Inductive graph embedding:</b> embed nodes never seen during training from their attributes "
        "and neighbourhood, without retraining.",
        x,
        top,
        width,
        BODY,
    )
    top -= 0.3 * inch
    inner = 0.28 * inch
    para = Paragraph(
        "<font color='#007C83'><b>Question.</b></font> Can a closed-form, deterministic method use graph "
        "structure well enough to compete with trained GNN and graph-SSL encoders?",
        BODY,
    )
    _, h = para.wrap(width - 2 * inner, PAGE_H)
    box(c, x, top - h - 2 * inner, width, h + 2 * inner, TEAL_LIGHT, TEAL, radius=10)
    para.drawOn(c, x + inner, top - inner - h)
    top -= h + 2 * inner + 0.3 * inch
    top = paragraph(
        c,
        "<b>Strict inductive protocol.</b> The encoder is built on the training subgraph; test nodes and "
        "their edges are seen only at inference.",
        x,
        top,
        width,
        BODY,
    )
    top -= 0.22 * inch
    for item in [
        "<b>5 benchmarks:</b> ogbn-arxiv, Reddit, Yelp, ogbn-products, PPI",
        "<b>10 baselines:</b> supervised GNN, contrastive, masked-AE and linear",
        "<b>Ablation CCA*:</b> the same two views with <i>M</i>&nbsp;=&nbsp;<i>I</i>",
    ]:
        top = paragraph(c, f"<font color='#007C83'>•</font>&nbsp;&nbsp;{item}", x, top, width, BODY_SMALL)
        top -= 0.12 * inch
    return top


def draw_method_card(c, x, top, width):
    top = heading(c, "Closed-form method", x, top, width)
    top = paragraph(c, "Two deterministic views of the node attributes <b>X</b>:", x, top, width, BODY)
    top -= 0.2 * inch
    eq_h = 1.65 * inch
    box(c, x, top - eq_h, width, eq_h, NAVY, radius=10)
    cx = x + width / 2
    draw_math(c, r"X_1 = X,\qquad X_2 = \widehat{P}^{K} X", cx, top - 0.2 * inch, 38, "#FFFFFF")
    draw_math(c, r"M = \frac{1}{2}(P + P^{2}),\qquad P = D^{-1}A", cx, top - 0.95 * inch, 33, "#D7E8EC")
    top -= eq_h + 0.28 * inch

    steps = [
        ("Propagate.", "<i>K</i> sparse products give <i>X</i><sub>2</sub>; no dense graph powers."),
        ("Whiten.", "Cholesky-factor the ridged <i>p</i> × <i>p</i> covariances."),
        ("Solve once.", "One truncated rank-<i>d</i> SVD:"),
        ("Infer.", "Unseen nodes are embedded by fixed projections:"),
    ]
    formulas = {
        2: r"\mathrm{SVD}_d\left(C_{11}^{-1/2}\, X_1^{\top} M X_2\, C_{22}^{-1/2}\right)",
        3: r"Z' = [\, X'\widetilde{W} \;\Vert\; \widehat{P}'^{K} X' \widetilde{H} \,]",
    }
    text_x = x + 0.62 * inch
    for i, (label, detail) in enumerate(steps):
        c.setFillColor(TEAL)
        c.circle(x + 0.25 * inch, top - 0.2 * inch, 0.23 * inch, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(x + 0.25 * inch, top - 0.28 * inch, str(i + 1))
        top = paragraph(c, f"<b>{label}</b> {detail}", text_x, top, x + width - text_x, BODY_SMALL)
        if i in formulas:
            top = draw_math(c, formulas[i], (text_x + x + width) / 2, top - 0.08 * inch, 27)
        top -= 0.18 * inch

    top -= 0.05 * inch
    cx_h = 1.3 * inch
    box(c, x, top - cx_h, width, cx_h, GOLD_LIGHT, GOLD, radius=10)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 23)
    c.drawString(x + 0.28 * inch, top - 0.45 * inch, "Cost")
    c.setFont("Helvetica", 21)
    c.setFillColor(MUTED)
    c.drawRightString(x + width - 0.28 * inch, top - 0.45 * inch, "no GPU · no seeds · no augmentations")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 31)
    c.drawCentredString(x + width / 2, top - 1.04 * inch, "O(K|E|p + np² + p³)")
    return top - cx_h


def draw_ablation_card(c, x, top, width):
    top = heading(c, "What does M buy?", x, top, width)
    top = paragraph(
        c,
        "Gain of GW-CCA over CCA* (same views, <i>M</i> = <i>I</i>), percentage points:",
        x,
        top,
        width,
        BODY,
    )
    top -= 0.25 * inch

    # GW-CCA minus CCA*, from Tables 1 and 2.
    nc = [0.2, 0.9, 11.4, -0.1, 7.1]
    lp = [2.8, 2.5, 7.4, 1.9, 15.5]
    label_w = 2.35 * inch
    value_w = 0.9 * inch
    zero_x = x + label_w + 0.1 * inch
    scale = (x + width - value_w - zero_x) / 16.0
    bar_h = 0.3 * inch
    group_h = 2 * bar_h + 0.17 * inch
    c.setFont("Helvetica-Bold", 22)
    for i, name in enumerate(DATASETS):
        gy = top - i * group_h
        c.setFillColor(INK)
        c.drawRightString(x + label_w, gy - bar_h - 0.08 * inch, name)
        for j, (value, color) in enumerate([(nc[i], TEAL), (lp[i], GOLD)]):
            by = gy - (j + 1) * bar_h
            c.setFillColor(color)
            length = value * scale
            c.rect(min(zero_x, zero_x + length), by + 0.03 * inch, abs(length), bar_h - 0.06 * inch, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold" if value >= 5 else "Helvetica", 20)
            label = f"+{value:.1f}" if value >= 0 else f"–{-value:.1f}"
            c.drawString(zero_x + max(length, 0) + 0.08 * inch, by + 0.07 * inch, label)
        c.setFont("Helvetica-Bold", 22)
    chart_bottom = top - 5 * group_h + 0.15 * inch
    c.setStrokeColor(INK)
    c.setLineWidth(1.5)
    c.line(zero_x, top + 0.05 * inch, zero_x, chart_bottom)

    legend_y = chart_bottom - 0.42 * inch
    lx = x + 0.1 * inch
    for color, label in [(TEAL, "Node classification"), (GOLD, "Link prediction (AUC)")]:
        c.setFillColor(color)
        c.rect(lx, legend_y, 0.3 * inch, 0.24 * inch, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica", 20)
        c.drawString(lx + 0.4 * inch, legend_y + 0.02 * inch, label)
        lx += 0.4 * inch + stringWidth(label, "Helvetica", 20) + 0.45 * inch
    top = legend_y - 0.3 * inch
    top = paragraph(
        c,
        "Largest gains on pairwise link prediction, where <i>M</i> weights exactly the "
        "node pairs the task scores.",
        x,
        top,
        width,
        BODY_SMALL,
    )
    top -= 0.25 * inch
    inner = 0.24 * inch
    para = Paragraph(
        "<b>Limitation.</b> Trails GraphSAGE on PPI node classification (55.3 vs 62.8): its small graphs "
        "favour deep message passing over shallow propagation.",
        BODY_SMALL,
    )
    _, h = para.wrap(width - 2 * inner, PAGE_H)
    box(c, x, top - h - 2 * inner, width, h + 2 * inner, SOFT, LINE, radius=10)
    para.drawOn(c, x + inner, top - inner - h)
    return top - h - 2 * inner


def draw_cards(c, top, height):
    col_w = (CONTENT_W - 2 * GAP) / 3
    pad = 0.36 * inch
    bottom = top - height
    for i, draw in enumerate([draw_problem_card, draw_method_card, draw_ablation_card]):
        x = MARGIN + i * (col_w + GAP)
        box(c, x, bottom, col_w, height, white, LINE, radius=14)
        end = draw(c, x + pad, top - pad, col_w - 2 * pad)
        slack = (end - bottom - pad) / inch
        print(f"{draw.__name__}: {slack:+.2f} in slack")
        assert slack >= 0, f"{draw.__name__} overflows its card by {-slack:.2f} in"
    return bottom


RESULT_HEADER_H, RESULT_GROUP_H, RESULT_ROW_H = 0.7 * inch, 0.56 * inch, 0.6 * inch
RESULTS_H = 1.0 * inch + RESULT_HEADER_H + 3 * RESULT_GROUP_H + 6 * RESULT_ROW_H + 0.95 * inch


def draw_results(c, top):
    # Strongest baseline per column excludes the CCA* ablation (paper Tables 1-3).
    nc_base = ["69.5", "94.6", "57.1", "74.3", "62.8"]
    nc_base_names = ["SGC", "GraphMAE2", "GraphMAE2", "GraphMAE2", "GraphSAGE"]
    nc_gw = ["69.7", "96.3", "61.1", "72.9", "55.3"]
    lp_base = ["95.3", "94.9", "93.0", "98.1", "87.5"]
    lp_base_names = ["GraphACL", "GraphSAGE", "GRACE", "GGD", "GRACE"]
    lp_gw = ["97.5", "97.6", "93.7", "99.5", "89.2"]
    t_base = ["11", "125", "226", "139", "11"]
    t_base_names = ["CCA-SSG", "GraphACL", "GGD", "GraphACL", "CCA-SSG"]
    t_gw = ["1.8", "42.0", "28.1", "12.1", "0.3"]

    header_h, group_h, row_h = RESULT_HEADER_H, RESULT_GROUP_H, RESULT_ROW_H
    pad = 0.4 * inch
    y = top - RESULTS_H
    box(c, MARGIN, y, CONTENT_W, RESULTS_H, white, LINE, radius=16)
    x = MARGIN + pad
    width = CONTENT_W - 2 * pad
    heading(c, "Benchmark results", x, top - pad + 0.08 * inch, width)
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 27)
    c.drawRightString(
        x + width, top - pad + 0.08 * inch - 34, "Best on 3/5 node-classification and 5/5 link-prediction benchmarks"
    )

    label_w = 5.1 * inch
    col_w = (width - label_w) / 5
    cursor = top - 1.0 * inch

    def cell_x(i):
        return x + label_w + i * col_w

    # Header row.
    cursor -= header_h
    c.setFillColor(NAVY)
    c.rect(x, cursor, width, header_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(x + 0.2 * inch, cursor + 0.24 * inch, "Method")
    for i, name in enumerate(DATASETS):
        c.drawCentredString(cell_x(i) + col_w / 2, cursor + 0.24 * inch, name)

    def group(title, note):
        nonlocal cursor
        cursor -= group_h
        c.setFillColor(SOFT)
        c.rect(x, cursor, width, group_h, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 25)
        c.drawString(x + 0.2 * inch, cursor + 0.18 * inch, title)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 21)
        c.drawString(x + 0.3 * inch + stringWidth(title, "Helvetica-Bold", 25), cursor + 0.18 * inch, note)

    def row(label, values, names=None, bold=(), highlight=False):
        nonlocal cursor
        cursor -= row_h
        if highlight:
            c.setFillColor(TEAL_LIGHT)
            c.rect(x, cursor, width, row_h, fill=1, stroke=0)
        c.setStrokeColor(LINE)
        c.setLineWidth(1)
        c.line(x, cursor, x + width, cursor)
        paragraph(c, label, x + 0.2 * inch, cursor + row_h - 0.13 * inch, label_w, ParagraphStyle(
            "rowlabel", parent=CELL, alignment=TA_LEFT, fontName="Helvetica-Bold" if highlight else "Helvetica"))
        for i, value in enumerate(values):
            text = f"<b>{value}</b>" if i in bold else value
            if names:
                text += f"<font size='18' color='#4B5E6B'>&nbsp;&nbsp;{names[i]}</font>"
            paragraph(c, text, cell_x(i), cursor + row_h - 0.13 * inch, col_w, CELL)

    group("Node classification", "   accuracy / Micro-F1 (%)")
    row("Strongest baseline", nc_base, nc_base_names, bold={3, 4})
    row("GW-CCA", nc_gw, bold={0, 1, 2}, highlight=True)
    group("Link prediction", "   AUC (%)")
    row("Strongest baseline", lp_base, lp_base_names)
    row("GW-CCA", lp_gw, bold=set(range(5)), highlight=True)
    group("Encoder construction time", "   seconds; GW-CCA on CPU, deep baselines on one A10G GPU")
    row("Fastest deep baseline", t_base, t_base_names)
    row("GW-CCA  (CPU)", t_gw, bold=set(range(5)), highlight=True)

    paragraph(
        c,
        "Strongest baseline = best of the 10 baselines per column, excluding the CCA* ablation (deep baselines: mean over 5 seeds; GW-CCA is "
        "deterministic). Bold = best in column. Timings are end-to-end but not hardware-matched (CPU vs GPU). "
        "Full tables with standard deviations: paper Tables 1–3.",
        x,
        cursor - 0.2 * inch,
        width,
        NOTE,
    )
    return y


def draw_footer(c, height):
    y = MARGIN
    box(c, MARGIN, y, CONTENT_W, height, NAVY, radius=16)
    pad = 0.42 * inch
    qr_size = height - 2 * 0.3 * inch
    qr_x = PAGE_W - MARGIN - 0.3 * inch - qr_size
    text_w = qr_x - (MARGIN + pad) - 0.4 * inch

    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(MARGIN + pad, y + height - 0.6 * inch, "TAKEAWAY")
    paragraph(
        c,
        "Graph-weighting the cross-covariance turns CCA into a competitive inductive graph encoder: "
        "one closed-form SVD, no iterative training.",
        MARGIN + pad,
        y + height - 0.72 * inch,
        text_w,
        ParagraphStyle("takeaway", parent=BODY, fontName="Helvetica-Bold", fontSize=32, leading=39, textColor=white),
    )
    c.setFillColor(NAVY_TEXT)
    c.setFont("Helvetica", 25)
    c.drawString(MARGIN + pad, y + 0.34 * inch, "Code: github.com/jianjinuta/gw-cca")
    c.drawRightString(MARGIN + pad + text_w, y + 0.34 * inch, "Contact: jensendengjj@gmail.com")

    box(c, qr_x, y + 0.3 * inch, qr_size, qr_size, white, radius=8)
    widget = QrCodeWidget(CODE_URL, barBorder=2)
    x0, y0, x1, y1 = widget.getBounds()
    drawing = Drawing(qr_size, qr_size, transform=[qr_size / (x1 - x0), 0, 0, qr_size / (y1 - y0), 0, 0])
    drawing.add(widget)
    renderPDF.draw(drawing, c, qr_x, y + 0.3 * inch)
    c.linkURL(CODE_URL, (qr_x, y + 0.3 * inch, qr_x + qr_size, y + 0.3 * inch + qr_size))
    return y + height


def main():
    plt.rcParams.update({"mathtext.fontset": "cm", "pdf.fonttype": 42})
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Graph-Weighted Canonical Correlation Analysis — LoG 2026 poster")
    c.setAuthor("Jianjin Deng and Di Ming")
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    top = draw_header(c) - GAP
    top = draw_figure1(c, top) - GAP
    footer_top = draw_footer(c, 2.5 * inch)
    # Results sit directly above the footer; the cards take the remaining height.
    results_top = footer_top + GAP + RESULTS_H
    draw_results(c, results_top)
    cards_h = top - (results_top + GAP)
    print(f"cards height: {cards_h / inch:.2f} in")
    draw_cards(c, top, cards_h)

    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
