"""
capture_screens.py: automatic screenshots of the app for notebook 05 and the slides (P5).

Requirements (once):  pip install playwright  &&  python -m playwright install chromium
Usage:
    1. in one terminal:  streamlit run app/streamlit_app.py
    2. in another one:   python -m scripts.capture_screens [--url http://localhost:8501] [--persona "Dark academia reader"]
The images are written to docs/figures/05_screen_*.png.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from urllib.parse import quote

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
TABS = ["Your profile", "Compare", "Semantic map", "Under the hood"]


async def capture(url: str, persona: str, height: int) -> None:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": height})
        # The ?persona=... parameter pre-fills the questionnaire (see streamlit_app.py)
        await page.goto(f"{url}/?persona={quote(persona)}", wait_until="networkidle")
        await page.wait_for_timeout(4000)
        steps = ["1_free_text", "2_genres_themes", "3_context", "4_likert", "5_guided"]
        for i, name in enumerate(steps):
            await page.screenshot(path=str(OUT / f"05_screen_{name}.png"))
            if i < len(steps) - 1:
                await page.get_by_role("button", name="Next →").click()
                await page.wait_for_timeout(1500)
        await page.get_by_role("button", name="See my recommendations ✨").click()
        await page.wait_for_selector("text=Your recommendations", timeout=600_000)
        await page.wait_for_timeout(6000)
        await page.screenshot(path=str(OUT / "05_screen_6_top3.png"))
        for tab in TABS:
            await page.get_by_role("tab", name=tab).click()
            await page.wait_for_timeout(4000)
            await page.screenshot(path=str(OUT / f"05_screen_7_{tab.lower().replace(' ', '_')}.png"))
        await browser.close()
    print(f"Screenshots written to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--persona", default="Dark academia reader")
    parser.add_argument("--height", type=int, default=1600)
    args = parser.parse_args()
    asyncio.run(capture(args.url, args.persona, args.height))
