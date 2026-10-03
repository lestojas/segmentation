#!/bin/bash
LABEL="com.user.dfa-slot-monitor"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Removed the scheduled job. (Files in this folder are untouched.)"
