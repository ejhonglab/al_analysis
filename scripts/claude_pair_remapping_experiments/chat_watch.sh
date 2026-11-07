#!/bin/bash
# Wait for a new chat.md entry that THIS SESSION did not write, then print it and exit (run as a
# background command so its exit wakes the agent). Entries start with "## <date> <time> <host>";
# entries this session posted with chat_post.sh (fingerprints in its ledger,
# ~/.local/state/kc_chat/<session_id>.posted) are ignored, so posting does not wake the poster,
# while entries from other hosts and from other sessions on this host do wake it. inotify does not
# work over NFS, hence polling (every 60 s by default). Only queue workers run this.
# Each poll also refreshes this session's heartbeat for the queue monitor
# (presence/<host>__<session_id>.watch next to chat.md; see monitor-server/), which is how the
# monitor tells a session idling on the queue from one that is gone.
# Usage: scripts/claude_pair_remapping_experiments/chat_watch.sh [CHAT_MD] [INTERVAL_S]
#        (CHAT_MD default: $KC_CHAT_MD, else /local/tensor/Tom/claude_kc_remapping/chat.md)
f=${1:-${KC_CHAT_MD:-/local/tensor/Tom/claude_kc_remapping/chat.md}}
interval=${2:-60}
me=$(hostname -s)
here=$(cd "$(dirname "$0")" && pwd)

sid=${CLAUDE_CODE_SESSION_ID:-}
[[ $sid =~ ^[A-Za-z0-9._-]{1,128}$ ]] || sid=pid${CLAUDE_PID:-$$}
ledger=${XDG_STATE_HOME:-$HOME/.local/state}/kc_chat/$sid.posted
presence=$(dirname "$f")/presence
hb=$presence/${me}__${sid}.watch
heartbeat() {  # $1: listening | exited. Bounded by timeout, since the NAS is a hard NFS mount.
  local json
  json=$(printf '{"host": "%s", "session_id": "%s", "claude_pid": %s, "watch_pid": %s, "state": "%s", "time": "%s", "epoch": %s, "interval_s": %s}' \
    "$me" "$sid" "${CLAUDE_PID:-null}" "$$" "$1" "$(date -Iseconds)" "$(date +%s)" "$interval")
  timeout -k 2 10 sh -c 'mkdir -p "$1" && printf "%s\n" "$2" > "$3.tmp.$$" && mv -f "$3.tmp.$$" "$3"' \
    _ "$presence" "$json" "$hb" 2>/dev/null
}
trap 'heartbeat exited' EXIT
trap 'exit 143' TERM INT HUP

n0=$(python3 -I "$here/chat_entries.py" count "$f")
while true; do
  heartbeat listening
  sleep "$interval" & wait $!
  if new=$(timeout -k 2 30 python3 -I "$here/chat_entries.py" new "$f" "$n0" "$ledger"); then
    echo "new chat.md entries (not from this session) at $(date '+%H:%M'):"
    printf '%s\n' "$new"
    exit 0
  fi
done
