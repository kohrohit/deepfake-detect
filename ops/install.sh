#!/usr/bin/env bash
# Install the dfd service as a systemd USER unit, so it starts automatically.
#
# No root, no system-wide install: this service binds 127.0.0.1 and writes
# under ~/.local/share/dfd. Run it from anywhere; it resolves the repository
# from its own location.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/dfd.service"

if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemctl not found. Run the service directly instead:" >&2
    echo "    cd $REPO && $PYTHON -m dfd.service -v" >&2
    exit 1
fi

echo "repository: $REPO"
echo "python:     $PYTHON"

"$PYTHON" -c "import dfd" 2>/dev/null || {
    echo "dfd is not importable by $PYTHON. Install it first:" >&2
    echo "    cd $REPO && $PYTHON -m pip install -e ." >&2
    exit 1
}

mkdir -p "$UNIT_DIR" "$HOME/.local/share/dfd/inbox"
sed -e "s|__REPO__|$REPO|g" -e "s|__PYTHON__|$PYTHON|g" \
    "$REPO/ops/dfd.service" > "$UNIT"
echo "wrote $UNIT"

systemctl --user daemon-reload
systemctl --user enable --now dfd.service
echo

# Lingering is what makes "automatically" true across a reboot. It needs a
# polkit prompt or root, so it is offered rather than assumed.
if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
    echo "To keep the service running after logout and across reboots:"
    echo "    sudo loginctl enable-linger $USER"
    echo
fi

sleep 2
systemctl --user --no-pager --lines=10 status dfd.service || true
echo
echo "dashboard: http://127.0.0.1:8077"
echo "inbox:     $HOME/.local/share/dfd/inbox   (drop files here)"
echo "logs:      journalctl --user -u dfd -f"
