# DFA slot monitor (Davao / Tagum, Oct–Dec)

Checks the DFA passport appointment calendar every 10 minutes and alerts you (push, email, Telegram, SMS and/or a Mac banner) when a date opens in **Davao** or **Tagum** for **October–December**. It only alerts: it never books for you. Alerts fire on *new* openings, so you won't get spammed.

## Install on your Mac (about 5 minutes)

```bash
git clone <this repo> && cd segmentation && git checkout matanglawin-dataset
cd dfa-slot-monitor
./install_mac.sh
```

The installer creates a Python environment, installs the browser, opens `.env` for you to fill in, sends a test alert, then schedules the job with launchd. Easiest alert setup: install the **ntfy** app on your phone, subscribe to a long random topic name, and put it in `NTFY_TOPIC=`. Add Gmail or SMS as a backup.

## Important limits

- **The Mac must be on, awake, and online** for checks to run. A closed or sleeping laptop doesn't check. To keep it awake on charger: `sudo pmset -c sleep 0`. For true 24/7 coverage, run it on a cheap VPS or an always-on machine instead (same code).
- You get a **"still running" message daily** (after `HEARTBEAT_HOUR`) and a **"BROKEN" alert after 3 failed checks in a row**. Silence is never assumed to mean "no slots".
- Don't set the interval below 5 minutes (`INTERVAL=300 ./install_mac.sh`); hammering the site gets your IP blocked.

## First-run calibration (read this)

The DFA site could not be reached while this was built, so the calendar selectors are auto-detecting guesses, verified only against a local fake booking page (`tests/fake_site.html`). Do this once:

0. **Find the real site names first:** `.venv/bin/python check_slots.py --list-sites` prints every option the DFA site shows. Put the exact names for Davao and Tagum in `SITES=` in `.env` (each entry is a case-insensitive regex). The defaults are already set to the confirmed labels `DAVAO(SM CITY DAVAO)` and `TAGUM (ROBINSONS PLACE OF TAGUM)`; use this only to double-check or if DFA renames them. Format is `Label::regex`; the log prints which option was selected and warns if several match.
1. `.venv/bin/python check_slots.py --headed` and watch it. It should accept the terms, pick the site, and reach the calendar.
2. If it fails, look in `debug/` (screenshot + HTML saved on every failure). Then set `SEL_MONTH_LABEL`, `SEL_MONTH_NEXT`, `SEL_AVAILABLE_DAY` in `.env` (CSS selectors; find them with right-click > Inspect in Chrome, or run `.venv/bin/playwright codegen <START_URL>`).
3. Check `logs/monitor.log` for `selected site option: '...'` and confirm it is the Davao / Tagum office you want.
4. Sanity check: confirm a date you can see as open in your normal browser is reported as open (`logs/monitor.log` shows "N open date(s)").

If DFA shows a CAPTCHA or Cloudflare challenge, the checker will fail and tell you; it deliberately does not try to bypass those.

## Commands

| | |
|---|---|
| Watch the log | `tail -f logs/monitor.log` |
| Run a check now | `launchctl kickstart -k gui/$(id -u)/com.user.dfa-slot-monitor` |
| Test alerts | `.venv/bin/python check_slots.py --test-notify` |
| Stop and remove | `./uninstall_mac.sh` |
| Reset "already alerted" memory | delete `state.json` |
