#!/usr/bin/env python
"""Render docs/whitepaper.md -> whitepaper.html (native inline SVG) + whitepaper.pdf.

The HTML keeps native inline SVG (browsers render it perfectly). For the PDF, each SVG is
RASTERIZED to a PNG and embedded as <img> — weasyprint's SVG engine renders these charts
incompletely, but every PDF renderer handles raster images, so all graphs show up.
"""
import os, re, base64

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "whitepaper.md")
HTML = os.path.join(ROOT, "docs", "whitepaper.html")
PDF = os.path.join(ROOT, "docs", "whitepaper.pdf")

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; color: #1a1a1a;
       line-height: 1.5; max-width: 820px; margin: 32px auto; padding: 0 16px;
       font-size: 15px; }
h1 { font-size: 26px; line-height: 1.25; border-bottom: 3px solid #2c3e50;
     padding-bottom: 10px; color: #1a2a3a; margin-top: 0; }
h2 { font-size: 20px; color: #2c3e50; border-bottom: 1px solid #ccd; padding-bottom: 5px;
     margin-top: 34px; }
h3 { font-size: 16px; color: #34495e; margin-top: 22px; }
p, li { font-size: 15px; }
strong { color: #111; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; background: #f4f5f7;
       padding: 1px 5px; border-radius: 3px; font-size: 13px; color: #b3005a; }
pre { background: #f7f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px 14px;
      overflow-x: auto; font-size: 12.5px; line-height: 1.45; }
pre code { background: none; color: #24292e; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13.5px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #2c3e50; color: #fff; }
tr:nth-child(even) td { background: #f6f8fa; }
td[align="right"], th[align="right"] { text-align: right; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 28px 0; }
blockquote { border-left: 4px solid #2c3e50; margin: 14px 0; padding: 4px 16px;
             color: #333; background: #f6f8fa; font-style: italic; }
a { color: #0969da; }
img { max-width: 100%; height: auto; display: block; margin: 8px auto; }
"""

# glyphs cairosvg's default font lacks -> ASCII so they don't render as boxes in the PDF.
# NOTE: "←" must escape its "<" (a bare "<" in text breaks strict XML parsing).
_GLYPHS = {"→": "->", "←": "&lt;-", "↑": "^", "↓": "v", "↕": "|", "≫": ">>", "·": "-",
           "×": "x", "≈": "~", "≤": "&lt;=", "≥": ">="}


def _rasterize_svgs(html):
    """Replace every inline <svg>..</svg> with a rasterized PNG <img> (for the PDF)."""
    import cairosvg

    def repl(m):
        svg = m.group(0)
        for k, v in _GLYPHS.items():
            svg = svg.replace(k, v)
        vb = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', svg)
        w, h = (vb.group(1), vb.group(2)) if vb else ("620", "320")
        svg = re.sub(r'style="[^"]*"', "", svg, count=1)             # cairosvg sizing fix
        svg = svg.replace("<svg ", f'<svg width="{w}" height="{h}" ', 1)
        try:
            png = cairosvg.svg2png(bytestring=svg.encode(),
                                   output_width=int(float(w)) * 2, background_color="white")
        except Exception as e:
            print(f"  ! chart rasterize failed ({e}); keeping inline SVG")
            return m.group(0)
        b64 = base64.b64encode(png).decode()
        return f'<img src="data:image/png;base64,{b64}"/>'

    return re.sub(r"<svg.*?</svg>", repl, html, flags=re.S)


def _despan_svgs(html):
    """python-markdown sometimes injects <p>/</p> inside an inline <svg> block (breaks strict XML
    parsing for the rasterizer and is invalid inside SVG). Strip them from every <svg>..</svg>."""
    def repl(m):
        return m.group(0).replace("<p>", "").replace("</p>", "")
    return re.sub(r"<svg.*?</svg>", repl, html, flags=re.S)


def main():
    text = open(MD).read()
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    body = _despan_svgs(body)

    def wrap(b):
        return (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>Efficient Thinking — White Paper</title><style>{CSS}</style></head>"
                f"<body>{b}</body></html>")

    open(HTML, "w").write(wrap(body))                                # HTML: native SVG
    print(f"wrote {HTML} ({os.path.getsize(HTML)} bytes)")

    pdf_html = wrap(_rasterize_svgs(body))                           # PDF: rasterized charts
    try:
        from weasyprint import HTML as WHTML
        WHTML(string=pdf_html).write_pdf(PDF)
        print(f"wrote {PDF} via weasyprint ({os.path.getsize(PDF)} bytes)")
        return
    except Exception as e:
        print(f"weasyprint unavailable ({type(e).__name__}); using xhtml2pdf")
    from xhtml2pdf import pisa
    with open(PDF, "wb") as f:
        res = pisa.CreatePDF(pdf_html, dest=f)
    print(f"wrote {PDF} via xhtml2pdf (err={res.err}, {os.path.getsize(PDF)} bytes)")


if __name__ == "__main__":
    main()
