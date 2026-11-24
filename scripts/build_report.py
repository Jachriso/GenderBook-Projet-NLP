"""
build_report.py: build the PDF documentation from the Markdown sections of docs/report/.

    python -m scripts.build_report                 # -> docs/GenderBook_report.pdf
    python -m scripts.build_report --html-only     # only docs/GenderBook_report.html (to check the layout)

Steps:
    1. read docs/report/*.md in file order (00 introduction ... 12 appendix B);
    2. build a title page (project, course, team read from the table of section 1.5);
    3. pandoc converts everything to one self-contained HTML file (figures embedded, formulas as
       MathML, table of contents);
    4. Google Chrome (headless) prints this HTML to an A4 PDF with page numbers.

Requirements: pandoc (https://pandoc.org) and Google Chrome. Nothing else to install: no LaTeX.
Edit the Markdown sections, never the PDF, then run this script again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "report"
OUT_PDF = ROOT / "docs" / "GenderBook_report.pdf"
OUT_HTML = ROOT / "docs" / "GenderBook_report.html"

TITLE = "GenderBook"
SUBTITLE = "Semantic book recommendation from natural-language preferences"
COURSE = "Natural Language Processing project · ECE Paris, ING5 Data & AI · 2026-27"
ASSIGNMENT = ("Semantic Analysis and Intelligent Contextual Recommendation Systems "
              "· Theme 3: Book Recommendation Based on Literary Preferences")
INSTRUCTOR = "Sarah Malaeb"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",     # macOS
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",           # Windows
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",  # Linux
]

# Print style: A4, serif body text, page numbers, each top-level section on a new page.
CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm;
        @bottom-center { content: counter(page); font: 9pt 'Helvetica Neue', Arial, sans-serif; color: #666; } }
@page :first { @bottom-center { content: none; } }
html { font-size: 10.5pt; }
body { font-family: Charter, 'Bitstream Charter', Georgia, 'Times New Roman', serif; line-height: 1.45;
       color: #1a1a1a; max-width: none; margin: 0; padding: 0; text-align: justify; hyphens: auto; }
h1, h2, h3, h4 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #12304f; text-align: left;
                 break-after: avoid; line-height: 1.25; }
h1 { font-size: 19pt; margin: 0 0 10pt; padding-bottom: 5pt; border-bottom: 1.5pt solid #12304f;
     break-before: page; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
code { font-family: Menlo, Consolas, 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f2f4f7;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f6f8fa; border: 0.5pt solid #d9dee5; padding: 7pt 9pt; font-size: 8pt; line-height: 1.3;
      overflow: hidden; white-space: pre; break-inside: avoid; text-align: left; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 8.8pt; line-height: 1.3;
        break-inside: auto; text-align: left; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #12304f; color: #fff; font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 600;
     padding: 4pt 5pt; border: 0.5pt solid #12304f; }
td { padding: 3.5pt 5pt; border: 0.5pt solid #cdd3db; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f7f9fb; }
figure { margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 8.5pt; color: #4a5565;
             margin-top: 4pt; font-style: italic; }
blockquote { margin: 8pt 0; padding: 2pt 10pt; border-left: 2.5pt solid #9fb3c8; color: #333; }
math[display="block"] { margin: 6pt 0; font-size: 1.05em; }
a { color: #12304f; text-decoration: none; }
/* title page */
.titlepage { height: 245mm; display: flex; flex-direction: column; justify-content: center; text-align: left; }
.titlepage .kicker { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; letter-spacing: 1.5pt;
                     text-transform: uppercase; color: #4a5565; margin-bottom: 14pt; }
.titlepage .title { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 40pt; font-weight: 700;
                    color: #12304f; margin: 0; }
.titlepage .subtitle { font-size: 16pt; color: #333; margin: 6pt 0 26pt; }
.titlepage .rule { width: 60mm; border-top: 2pt solid #12304f; margin-bottom: 22pt; }
.titlepage .meta { font-size: 10.5pt; color: #333; margin: 3pt 0; }
.titlepage table { width: 70%; margin-top: 22pt; font-size: 10pt; }
.titlepage td { background: none !important; border: none; border-bottom: 0.5pt solid #d9dee5; padding: 4pt 2pt; }
.titlepage .date { margin-top: 28pt; font-size: 10pt; color: #4a5565; }
/* table of contents */
nav#TOC { break-after: page; }
nav#TOC::before { content: "Contents"; display: block; font-family: 'Helvetica Neue', Arial, sans-serif;
                  font-size: 19pt; font-weight: 700; color: #12304f; margin-bottom: 10pt;
                  border-bottom: 1.5pt solid #12304f; padding-bottom: 5pt; }
nav#TOC ul { list-style: none; padding-left: 0; margin: 0; }
nav#TOC > ul > li { margin-top: 6pt; font-weight: 600; font-family: 'Helvetica Neue', Arial, sans-serif; }
nav#TOC ul ul { padding-left: 14pt; font-weight: 400; font-family: Charter, Georgia, serif; font-size: 9.5pt; }
nav#TOC a { color: #1a1a1a; }
"""


def team_from_introduction(intro: str) -> list[tuple[str, str]]:
    """(member, role) rows of the team table of section 1.5."""
    rows = []
    in_team = False
    for line in intro.splitlines():
        if line.startswith("## 1.5"):
            in_team = True
        elif line.startswith("## ") and in_team:
            break
        elif in_team and line.startswith("|") and not line.startswith("|---") and "Member" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0]:
                rows.append((cells[0], cells[1]))
    return rows


def title_page(team: list[tuple[str, str]]) -> str:
    rows = "".join(f"<tr><td><strong>{html.escape(m)}</strong></td><td>{html.escape(r)}</td></tr>" for m, r in team)
    return f"""<section class="titlepage">
<div class="kicker">{html.escape(COURSE)}</div>
<h1 class="title" style="border:none; break-before:auto; padding:0">{TITLE}</h1>
<div class="subtitle">{html.escape(SUBTITLE)}</div>
<div class="rule"></div>
<div class="meta"><strong>Assignment:</strong> {html.escape(ASSIGNMENT)}</div>
<div class="meta"><strong>Instructor:</strong> {html.escape(INSTRUCTOR)}</div>
<table>{rows}</table>
<div class="date">Project report · {dt.date.today().strftime('%B %Y')}</div>
</section>
"""


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        path = shutil.which(c) or (c if Path(c).exists() else None)
        if path:
            return path
    return None


def build(html_only: bool = False) -> Path:
    if not shutil.which("pandoc"):
        sys.exit("pandoc is required: https://pandoc.org/installing.html")
    sections = sorted(REPORT_DIR.glob("*.md"))
    if not sections:
        sys.exit(f"No Markdown section found in {REPORT_DIR}")
    intro = sections[0].read_text(encoding="utf-8")

    # One Markdown document; HTML comments (working notes) are removed so that they never reach the PDF
    body = "\n\n".join(re.sub(r"<!--.*?-->", "", p.read_text(encoding="utf-8"), flags=re.S) for p in sections)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "report.md").write_text(body, encoding="utf-8")
        (tmp / "style.css").write_text(CSS, encoding="utf-8")
        (tmp / "before.html").write_text(title_page(team_from_introduction(intro)), encoding="utf-8")
        cmd = [
            "pandoc", str(tmp / "report.md"),
            "--from", "markdown+tex_math_dollars+pipe_tables+implicit_figures-smart",
            "--to", "html5", "--standalone", "--embed-resources",
            "--mathml", "--toc", "--toc-depth=2",
            "--css", str(tmp / "style.css"),
            "--include-before-body", str(tmp / "before.html"),
            "--resource-path", f"{REPORT_DIR}{':' if sys.platform != 'win32' else ';'}{ROOT}",
            "--metadata", f"pagetitle={TITLE}: project report",
            "--metadata", "lang=en",
            "-o", str(OUT_HTML),
        ]
        subprocess.run(cmd, check=True)
    print(f"HTML: {OUT_HTML}")
    if html_only:
        return OUT_HTML

    chrome = find_chrome()
    if not chrome:
        sys.exit("Google Chrome not found: open the HTML file and print it to PDF (A4) from a browser.")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    OUT_HTML.unlink()   # the PDF is the deliverable; rebuild the HTML with --html-only when needed
    print(f"PDF:  {OUT_PDF}")
    return OUT_PDF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html-only", action="store_true", help="stop after the HTML file")
    build(parser.parse_args().html_only)
