#!/usr/bin/env python
"""Render docs/whitepaper.md -> whitepaper.html (styled, self-contained) + whitepaper.pdf."""
import os, sys
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
.subtitle { color: #555; font-size: 15px; }
"""

def main():
    text = open(MD).read()
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Chess-Scaling White Paper</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    open(HTML, "w").write(html)
    print(f"wrote {HTML} ({os.path.getsize(HTML)} bytes)")

    # PDF: prefer weasyprint (full CSS), fall back to xhtml2pdf
    try:
        from weasyprint import HTML as WHTML
        WHTML(string=html).write_pdf(PDF)
        print(f"wrote {PDF} via weasyprint ({os.path.getsize(PDF)} bytes)")
        return
    except Exception as e:
        print(f"weasyprint unavailable ({type(e).__name__}); using xhtml2pdf")
    from xhtml2pdf import pisa
    with open(PDF, "wb") as f:
        res = pisa.CreatePDF(html, dest=f)
    print(f"wrote {PDF} via xhtml2pdf (err={res.err}, {os.path.getsize(PDF)} bytes)")


if __name__ == "__main__":
    main()
