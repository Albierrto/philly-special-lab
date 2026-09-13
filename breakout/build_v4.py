"""Copy the app template into output/v4.   python -m breakout.build_v4

The whole app (shell, pages, JavaScript) lives in one file, breakout/templates/lab.html. assemble.py fills its
__DATA__ slot for the artifact editions and site.py wires it to data/*.js for the website.
"""
from __future__ import annotations
import sys
from pathlib import Path
from . import config as C


def main(argv=None):
    src = Path(__file__).with_name("templates") / "lab.html"
    html = src.read_text(encoding="utf-8")
    out = C.OUT / "v4"; out.mkdir(exist_ok=True); (out / "explorer_template.html").write_text(html, encoding="utf-8")
    print("wrote", out / "explorer_template.html", len(html))
    return 0


if __name__ == "__main__":
    sys.exit(main())
