"""Render reports/report.md -> report.html -> report.pdf (mistune ships with nbconvert; PDF via headless Chrome)."""
import re
import shutil
import subprocess
from pathlib import Path

import mistune

HERE = Path(__file__).parent
CSS = """body{font-family:Georgia,serif;max-width:820px;margin:auto;font-size:11pt;line-height:1.45;color:#111}
h1{font-size:17pt;border-bottom:1px solid #ccc;margin-top:1.4em}h2{font-size:13pt}
table{border-collapse:collapse;margin:.8em 0;font-size:9.5pt}th,td{border:1px solid #bbb;padding:3px 7px}
th{background:#f1f0ec}img{max-width:100%;display:block;margin:.6em auto}pre{background:#f6f6f4;padding:8px;font-size:9pt}
blockquote{border-left:3px solid #eb6834;margin-left:0;padding-left:12px;color:#333}
.title{text-align:center;font-size:22pt;margin-top:0}.sub{text-align:center;color:#555}
@page{size:A4;margin:16mm}"""


def main():
    text = (HERE / "report.md").read_text()
    meta = dict(re.findall(r'^(\w+): "(.*)"$', text.split("---")[1], re.M))
    body = mistune.create_markdown(plugins=["table"])(text.split("---", 2)[2])
    html = (f"<!doctype html><meta charset='utf-8'><title>{meta['title']}</title><style>{CSS}</style>"
            f"<p class='title'>{meta['title']}</p><p class='sub'>{meta['subtitle']} · {meta['date']}</p>{body}")
    (HERE / "report.html").write_text(html)
    chrome = shutil.which("google-chrome-stable") or shutil.which("chromium") or shutil.which("google-chrome")
    if chrome:
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={HERE / 'report.pdf'}", (HERE / "report.html").as_uri()],
                       check=True, capture_output=True)
        print("wrote", HERE / "report.pdf")
    else:
        print("Chrome not found: open reports/report.html and print to PDF")


if __name__ == "__main__":
    main()
