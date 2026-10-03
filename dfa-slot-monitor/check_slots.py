#!/usr/bin/env python3
"""Watch the DFA passport appointment calendar for open slots and notify you.

Run once per invocation (scheduled by launchd/cron). Alerts only on NEW availability.

    python check_slots.py                # normal run
    python check_slots.py --test-notify  # send a test message through every configured channel
    python check_slots.py --headed       # watch the browser work (for debugging selectors)
"""
import argparse
import calendar
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Error as PWError
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

import notify

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "state.json"
DEBUG_DIR = HERE / "debug"

# Defaults cover common calendar widgets (bootstrap-datepicker, jQuery UI, FullCalendar, react-datepicker).
DEFAULT_MONTH_LABEL = (".datepicker-switch, .ui-datepicker-title, .fc-toolbar-title, "
                       ".react-datepicker__current-month, .calendar-title, .month-title, [class*='month-year' i]")
DEFAULT_MONTH_NEXT = (".datepicker .next, .ui-datepicker-next, .fc-next-button, "
                      ".react-datepicker__navigation--next, button[aria-label*='next' i], [class*='next-month' i]")
DEFAULT_AVAILABLE = (".datepicker td.day:not(.disabled):not(.old):not(.new), "
                     "td[data-handler='selectDay']:not(.ui-state-disabled), "
                     ".fc-day.available, .react-datepicker__day:not(.react-datepicker__day--disabled):not(.react-datepicker__day--outside-month), "
                     "td.available:not(.disabled):not(.unavailable), .day.available, [data-available='true']")

# Exact dropdown labels on the DFA site: "DAVAO(SM CITY DAVAO)" and "TAGUM (ROBINSONS PLACE OF TAGUM".
# Patterns anchor on the distinctive start so spacing / a missing ")" can't break the match.
DEFAULT_SITES = r"Davao (SM City Davao)::^\s*DAVAO\s*\(\s*SM CITY,Tagum (Robinsons Place Tagum)::^\s*TAGUM\s*\(\s*ROBINSONS"

CHROME_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def load_env():
    """Tiny .env loader (no dependency). Real environment variables win."""
    f = HERE / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip("'\"")
        os.environ.setdefault(k.strip(), v)


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def csv(name, default):
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def parse_sites(raw):
    """'Label::regex,Label2::regex2' or plain regexes -> [(label, regex)]."""
    out = []
    for entry in [x.strip() for x in raw.split(",") if x.strip()]:
        label, _, pat = entry.partition("::")
        out.append((label.strip(), pat.strip()) if pat else (entry, entry))
    return out


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:  # noqa: BLE001
        return {"alerted": [], "failures": 0, "heartbeat_date": "", "last_ok": ""}


def save_state(st):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def save_debug(page, tag):
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        page.screenshot(path=str(DEBUG_DIR / f"{stamp}-{tag}.png"), full_page=True)
        (DEBUG_DIR / f"{stamp}-{tag}.html").write_text(page.content())
        # keep only the newest 40 files
        files = sorted(DEBUG_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
        for old in files[:-40]:
            old.unlink()
    except Exception as e:  # noqa: BLE001
        log(f"could not save debug info: {e}")


# --------------------------------------------------------------------------- site navigation

def click_next(page, pattern):
    """Tick consent checkboxes and press the first visible Next/Continue-style control."""
    for cb in page.locator("input[type=checkbox]:visible").all():
        try:
            if not cb.is_checked():
                cb.check(force=True)
        except PWError:
            pass
    rx = re.compile(pattern, re.I)
    for role in ("button", "link"):
        loc = page.get_by_role(role, name=rx)
        for i in range(loc.count()):
            el = loc.nth(i)
            if el.is_visible() and el.is_enabled():
                el.click()
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_timeout(800)
                return True
    return False


def select_site(page, site):
    """Pick `site` from any dropdown (handles cascading region>province>site) or clickable text. True if found."""
    rx = re.compile(site, re.I)
    chosen = False
    for _ in range(3):  # cascading dropdowns reveal further options after each choice
        progressed = False
        for sel in page.locator("select:visible").all():
            opts = sel.locator("option").all_inner_texts()
            hits = [(i, t.strip()) for i, t in enumerate(opts) if rx.search(t)]
            if len(hits) > 1:
                log(f"WARNING: '{site}' matches {len(hits)} options {[t for _, t in hits]}; using the first. "
                    f"Make SITES more specific (run --list-sites to see exact names)")
            if hits:
                idx, text = hits[0]
                if not sel.evaluate("(el, i) => el.selectedIndex === i", idx):
                    sel.select_option(index=idx)
                    log(f"selected site option: '{text}' (pattern '{site}')")
                    page.wait_for_timeout(700)
                    progressed = True
                chosen = True
        if not progressed:
            break
    if chosen:
        return True
    cand = page.get_by_text(rx)
    for i in range(cand.count()):
        if cand.nth(i).is_visible():
            cand.nth(i).click()
            page.wait_for_timeout(700)
            return True
    return False


def visible(page, selector):
    try:
        loc = page.locator(selector)
        return loc.count() > 0 and loc.first.is_visible()
    except PWError:
        return False


def to_calendar(page, cfg, site):
    """Walk the wizard: accept terms -> choose site -> reach the calendar."""
    page.goto("about:blank")  # guarantees a fresh load, even for single-page apps
    page.goto(cfg["start_url"], wait_until="domcontentloaded")
    site_done = False
    for _ in range(10):
        if visible(page, cfg["sel_month_label"]):
            return
        if not site_done:
            site_done = select_site(page, site)
            if site_done and visible(page, cfg["sel_month_label"]):
                return
        if not click_next(page, cfg["next_text"]):
            page.wait_for_timeout(1000)
    raise RuntimeError(f"could not reach the calendar for site '{site}' "
                       f"(site found: {site_done}). See debug/ screenshot, then adjust .env selectors")


def list_sites(headed=False):
    """Walk the wizard and print every dropdown option / selectable text so SITES can be set to real names."""
    cfg = make_cfg()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        page = browser.new_context(user_agent=CHROME_UA, locale="en-PH", timezone_id="Asia/Manila").new_page()
        page.set_default_timeout(30000)
        page.goto(cfg["start_url"], wait_until="domcontentloaded")
        seen = False
        for _ in range(10):
            selects = page.locator("select:visible").all()
            for n, sel in enumerate(selects, 1):
                opts = [o.strip() for o in sel.locator("option").all_inner_texts() if o.strip()]
                print(f"\nDropdown #{n} ({len(opts)} options):")
                for o in opts:
                    print(f"   - {o}")
                seen = True
            if seen:
                break
            if not click_next(page, cfg["next_text"]):
                page.wait_for_timeout(1000)
        if not seen:
            save_debug(page, "list-sites")
            print("No dropdown found. Saved a screenshot/HTML to debug/. Sites may be radio buttons or a map; "
                  "open debug/*.png to read the names.")
        else:
            print("\nCopy the exact names you want into SITES= in .env (comma-separated; each entry is a "
                  "case-insensitive regex, e.g. SITES=DFA Davao \\(SM City,DFA Tagum).")
            print("NOTE: dropdowns may cascade (region > province > site); if you only see regions, pick one in "
                  "--headed mode and read the next list, or run with --headed and watch.")
        browser.close()


def month_label(page, cfg):
    return page.locator(cfg["sel_month_label"]).first.inner_text().strip()


def goto_month(page, cfg, month_name):
    short = month_name[:3]
    for _ in range(14):
        label = month_label(page, cfg)
        if re.search(month_name, label, re.I) or re.search(rf"\b{short}\b", label, re.I):
            if not cfg["year"] or cfg["year"] in label:
                return True
        nxt = page.locator(cfg["sel_month_next"])
        if nxt.count() == 0 or not nxt.first.is_visible():
            return False
        nxt.first.click()
        page.wait_for_timeout(500)
    return False


def available_days(page, cfg):
    days = []
    loc = page.locator(cfg["sel_available"])
    for i in range(loc.count()):
        el = loc.nth(i)
        if el.is_visible():
            m = re.search(r"\d+", el.inner_text())
            days.append(m.group(0) if m else "?")
    return days


def scan(page, cfg):
    found = []
    for site, pattern in cfg["sites"]:
        to_calendar(page, cfg, pattern)
        for month in cfg["months"]:
            if not goto_month(page, cfg, month):
                log(f"{site}: {month} not shown in calendar (not released yet?)")
                continue
            days = available_days(page, cfg)
            log(f"{site} / {month}: {len(days)} open date(s) {days}")
            if days:
                found.append((site, month, days))
    return found


# --------------------------------------------------------------------------- main

def make_cfg():
    return {
        "start_url": os.environ.get("START_URL", "https://www.passport.gov.ph/appointment"),
        "sites": parse_sites(os.environ.get("SITES", DEFAULT_SITES)),
        "months": csv("MONTHS", "October,November,December"),
        "year": os.environ.get("YEAR", "").strip(),
        "next_text": os.environ.get("NEXT_BUTTON_TEXT", r"^(next|continue|proceed|submit|agree|accept)\b"),
        "sel_month_label": os.environ.get("SEL_MONTH_LABEL") or DEFAULT_MONTH_LABEL,
        "sel_month_next": os.environ.get("SEL_MONTH_NEXT") or DEFAULT_MONTH_NEXT,
        "sel_available": os.environ.get("SEL_AVAILABLE_DAY") or DEFAULT_AVAILABLE,
    }


def run(headed=False):
    cfg = make_cfg()
    st = load_state()
    page = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headed)
            ctx = browser.new_context(user_agent=CHROME_UA, locale="en-PH", timezone_id="Asia/Manila",
                                      viewport={"width": 1280, "height": 1000})
            ctx.set_default_timeout(30000)
            page = ctx.new_page()
            try:
                found = scan(page, cfg)
            except Exception:
                save_debug(page, "error")
                raise
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001
        st["failures"] = st.get("failures", 0) + 1
        log(f"CHECK FAILED ({st['failures']} in a row): {e}")
        if st["failures"] == 3 or (st["failures"] > 3 and st["failures"] % 50 == 0):
            notify.send_all("DFA monitor is BROKEN",
                            f"{st['failures']} failed checks in a row, so you are NOT being watched.\n"
                            f"Last error: {str(e)[:300]}\nSee dfa-slot-monitor/debug/ on your Mac.")
        save_state(st)
        return 1

    st["failures"] = 0
    st["last_ok"] = datetime.now().isoformat(timespec="seconds")
    current = {f"{site}|{month}|{d}" for site, month, days in found for d in days}
    new_keys = current - set(st.get("alerted", []))
    if new_keys:
        lines = []
        for site, month, days in found:
            fresh = [d for d in days if f"{site}|{month}|{d}" in new_keys]
            if fresh:
                lines.append(f"{site} - {month}: {', '.join(fresh)}")
        notify.send_all("DFA slot available!", "\n".join(lines), cfg["start_url"])
        log("ALERT SENT: " + " | ".join(lines))
    st["alerted"] = sorted(current)  # slots that vanish are forgotten, so they re-alert if they return

    hb_hour = int(os.environ.get("HEARTBEAT_HOUR", "20"))
    today = datetime.now().strftime("%Y-%m-%d")
    if hb_hour >= 0 and datetime.now().hour >= hb_hour and st.get("heartbeat_date") != today:
        notify.send_all("DFA monitor: still running",
                        f"No action needed. Last good check {st['last_ok']}. "
                        f"Open dates right now: {len(current)}.")
        st["heartbeat_date"] = today
    save_state(st)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-notify", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--list-sites", action="store_true", help="print the real site names shown on the DFA site")
    args = ap.parse_args()
    load_env()
    if args.list_sites:
        list_sites(headed=args.headed)
        return 0
    if args.test_notify:
        ok = notify.send_all("DFA monitor test", "If you can read this, notifications work.")
        print("Delivered via:", ok or "NOTHING (check .env)")
        return 0 if ok else 1
    start = time.time()
    rc = run(headed=args.headed)
    log(f"done in {time.time() - start:.1f}s (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
