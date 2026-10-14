"""
Browser-level UI regression tests.

These exist because two unreadable-dashboard bugs shipped that unit tests could not
catch, both of them CSS/stacking problems:

  1. a filled mask over every person covered the video entirely;
  2. `.stage img { display: block }` silently defeated the `hidden` attribute, so the
     static keyframe sat on top of the playing video and the dashboard showed one
     frozen frame.

Both were invisible to the pytest suite and obvious to a human looking at the screen.
They are skipped automatically when the app or a browser is unavailable.
"""

from __future__ import annotations

import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE_URL = os.environ.get("WAYFINDER_UI_URL", "http://127.0.0.1:8220")

pytestmark = pytest.mark.skipif(
    os.environ.get("WAYFINDER_SKIP_UI_TESTS") == "1",
    reason="UI tests disabled via WAYFINDER_SKIP_UI_TESTS",
)


def _browser():
    playwright = pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright  # noqa: F401

    return playwright


@pytest.fixture(scope="module")
def page():
    _browser()
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path="/usr/bin/chromium", args=["--no-sandbox"]
            )
            context = browser.new_context(viewport={"width": 1400, "height": 860})
            pg = context.new_page()
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on(
                "console",
                lambda m: errors.append(m.text) if m.type == "error" else None,
            )
            pg.errors = errors
            pg.goto(BASE_URL, wait_until="networkidle", timeout=60000)
            pg.wait_for_timeout(9000)
            yield pg
            browser.close()
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"browser unavailable: {exc}")


def test_dashboard_loads_without_script_errors(page):
    assert not page.errors, f"page reported errors: {page.errors[:3]}"
    assert page.title().startswith("Wayfinder")


def test_video_layer_is_visible_and_snapshot_is_not(page):
    """Live mode must show the moving video, never the static keyframe on top of it."""
    state = page.evaluate(
        """() => {
          const v = document.getElementById('video');
          const i = document.getElementById('snapshot');
          return {
            video_hidden: v.hidden,
            video_display: getComputedStyle(v).display,
            video_z: parseInt(getComputedStyle(v).zIndex || '0', 10),
            img_hidden: i.hidden,
            img_display: getComputedStyle(i).display,
            img_z: parseInt(getComputedStyle(i).zIndex || '0', 10),
          };
        }"""
    )
    assert state["video_display"] != "none", "the video layer must be visible in live mode"
    assert state["img_display"] == "none", (
        "the static keyframe is visible in live mode; it will cover the playing video"
    )
    assert state["video_z"] > state["img_z"], (
        "the video must stack above the still frame so a stale frame cannot hide it"
    )


def test_documented_css_cannot_defeat_the_hidden_attribute(page):
    """
    `[hidden] { display: none !important }` must exist, because author rules like
    `.stage img { display: block }` otherwise win against the browser default and make
    `hidden` a no-op.
    """
    forced = page.evaluate(
        """() => {
          const probe = document.createElement('div');
          probe.hidden = true;
          probe.style.display = 'block';   // simulate an author rule
          document.body.appendChild(probe);
          const display = getComputedStyle(probe).display;
          probe.remove();
          return display;
        }"""
    )
    assert forced == "none", (
        f"[hidden] does not beat an author display rule (got '{forced}'); "
        "layer visibility toggles will silently stop working"
    )


def test_the_picture_actually_changes_over_time(page):
    """A single frozen frame is exactly the bug this guards against."""
    signatures = []
    for _ in range(4):
        page.wait_for_timeout(1400)
        signatures.append(hashlib.sha256(page.locator("#stage").screenshot()).hexdigest())
    assert len(set(signatures)) > 1, (
        "the video area rendered the same pixels four times; it is not playing"
    )


def test_snapshot_mode_shows_the_still_frame_instead(page):
    try:
        page.click("#mode-snap")
        page.wait_for_timeout(1800)
        state = page.evaluate(
            """() => ({
              video_display: getComputedStyle(document.getElementById('video')).display,
              img_display: getComputedStyle(document.getElementById('snapshot')).display,
              img_loaded: document.getElementById('snapshot').naturalWidth > 0,
            })"""
        )
        assert state["img_display"] != "none", "snapshot mode must show the still frame"
        assert state["video_display"] == "none", "the video should be hidden in snapshot mode"
        assert state["img_loaded"], "the snapshot image did not load"
    finally:
        page.click("#mode-live")
        page.wait_for_timeout(3000)


def test_overlay_is_drawn_and_does_not_blank_the_frame(page):
    """Boxes are drawn on the canvas; the canvas must not be an opaque fill."""
    page.wait_for_timeout(3000)
    ink = page.evaluate(
        """() => {
          const c = document.getElementById('overlay');
          const ctx = c.getContext('2d');
          const { data } = ctx.getImageData(0, 0, c.width, c.height);
          let painted = 0;
          for (let i = 3; i < data.length; i += 4) if (data[i] > 10) painted++;
          return { painted, total: data.length / 4 };
        }"""
    )
    assert ink["painted"] > 0, "no overlay was drawn"
    # Boxes and the zone outline only. A filled silhouette per person used to paint tens
    # of thousands of pixels and made the footage unreadable.
    ratio = ink["painted"] / ink["total"]
    assert ratio < 0.20, (
        f"the overlay covers {ratio:.0%} of the frame; it should only draw boxes and "
        "the waiting-zone outline"
    )
