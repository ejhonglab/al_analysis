#!/bin/bash
# Wait for a new chat.md entry from ANOTHER host, then print it and exit (run as a background
# command so its exit wakes the agent). Entries start with "## <date> <time> <host>"; entries this
# host appends are ignored, so posting to chat no longer wakes the poster. inotify does not work
# over NFS, hence polling (every 60 s by default).
# Usage: scripts/claude_pair_remapping_experiments/chat_watch.sh [CHAT_MD] [INTERVAL_S]
#        (CHAT_MD default: $KC_CHAT_MD, else /local/tensor/Tom/claude_kc_remapping/chat.md)
f=${1:-${KC_CHAT_MD:-/local/tensor/Tom/claude_kc_remapping/chat.md}}
interval=${2:-60}
me=$(hostname -s)
others() { grep -E '^## [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} ' "$f" | grep -cvE " ${me}[[:space:]]*$"; }
n0=$(others)
while true; do
  sleep "$interval"
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
