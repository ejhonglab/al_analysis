#!/bin/bash
# Wait for a new chat.md entry from ANOTHER host, then print it and exit (run as a background
# command so its exit wakes the agent). Entries start with "## <date> <time> <host>"; entries this
# host appends are ignored, so posting to chat no longer wakes the poster. inotify does not work
# over NFS, hence polling (every 60 s by default).
# Each poll also refreshes this session's heartbeat for the queue monitor
# (presence/<host>__<session_id>.watch next to chat.md; see monitor-server/), which is how the
# monitor tells a session idling on the queue from one that is gone.
# Usage: scripts/claude_pair_remapping_experiments/chat_watch.sh [CHAT_MD] [INTERVAL_S]
#        (CHAT_MD default: $KC_CHAT_MD, else /local/tensor/Tom/claude_kc_remapping/chat.md)
f=${1:-${KC_CHAT_MD:-/local/tensor/Tom/claude_kc_remapping/chat.md}}
interval=${2:-60}
me=$(hostname -s)
others() { grep -E '^## [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} ' "$f" | grep -cvE " ${me}[[:space:]]*$"; }

sid=${CLAUDE_CODE_SESSION_ID:-}
[[ $sid =~ ^[A-Za-z0-9._-]{1,128}$ ]] || sid=pid${CLAUDE_PID:-$$}
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

n0=$(others)
while true; do
  heartbeat listening
  sleep "$interval" & wait $!
  n1=$(others)
  if [ "$n1" -gt "$n0" ]; then
    echo "new chat.md entries from other hosts at $(date '+%H:%M'):"
    # print every entry after the last one seen from another host
    awk -v me="$me" -v skip="$n0" '
      /^## [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} / {
        host = $NF; if (host != me) { k++ } ; show = (k > skip) }
      show { print }' "$f"
    exit 0
  fi
done
