#!/usr/bin/env bash
# Stop and remove the dfd user service. Leaves the database and workdir
# alone: they are the audit trail, and removing a unit file is not a reason
# to destroy records of decisions already made.
set -euo pipefail
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
systemctl --user disable --now dfd.service 2>/dev/null || true
rm -f "$UNIT_DIR/dfd.service"
systemctl --user daemon-reload
echo "removed $UNIT_DIR/dfd.service"
echo "kept ~/.local/share/dfd (database, inbox, workdir)"
