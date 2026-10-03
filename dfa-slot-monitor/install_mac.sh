#!/bin/bash
# One-command installer for macOS: venv + Playwright + a launchd job that runs every 10 minutes.
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="com.user.dfa-slot-monitor"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INTERVAL="${INTERVAL:-600}"   # seconds between checks; don't go below ~300

command -v python3 >/dev/null || { echo "python3 not found. Install it: xcode-select --install  (or brew install python)"; exit 1; }

echo "==> Creating virtualenv and installing dependencies"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/playwright install chromium

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "==> Created .env. Opening it now: fill in at least ONE notification channel (NTFY_TOPIC is easiest)."
  open -t .env 2>/dev/null || true
  read -r -p "Press Enter once you have saved .env... " _
fi

echo "==> Testing notifications"
.venv/bin/python check_slots.py --test-notify || echo "!! No channel delivered. Fix .env and re-run ./install_mac.sh"

mkdir -p logs "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$DIR/.venv/bin/python</string><string>$DIR/check_slots.py</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$DIR/logs/monitor.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/monitor.log</string>
</dict></plist>
PL

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo
echo "==> Installed. Checking every $((INTERVAL/60)) minutes. Log: $DIR/logs/monitor.log"
echo "    Watch it:      tail -f $DIR/logs/monitor.log"
echo "    Run now:       launchctl kickstart -k gui/$(id -u)/$LABEL"
echo "    Stop/remove:   ./uninstall_mac.sh"
echo
echo "IMPORTANT: it only runs while this Mac is awake and online."
echo "  Keep it awake on charger:  sudo pmset -c sleep 0   (Mac stays awake on charger; screen may still sleep)"
