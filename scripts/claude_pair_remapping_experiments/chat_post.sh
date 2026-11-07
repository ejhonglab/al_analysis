#!/bin/bash
# Append an entry to chat.md as this host, and record its fingerprint in this session's ledger so
# this session's chat_watch.sh does not wake on its own post (it still wakes on entries from other
# hosts and from other sessions on this host). Queue workers post with this, not with `cat >>`.
# Usage: scripts/claude_pair_remapping_experiments/chat_post.sh [CHAT_MD] <<'EOF'
#        <entry text, without the "## <date> <time> <host>" header, which this adds>
#        EOF
here=$(cd "$(dirname "$0")" && pwd)
default_chat() {  # $KC_CHAT_MD, else the first existing of the hosts' tensor mount points
  [ -n "${KC_CHAT_MD:-}" ] && { echo "$KC_CHAT_MD"; return; }
  for d in /local/tensor /mnt/tensor; do
    [ -f "$d/Tom/claude_kc_remapping/chat.md" ] && { echo "$d/Tom/claude_kc_remapping/chat.md"; return; }
  done
  echo /local/tensor/Tom/claude_kc_remapping/chat.md
}
f=${1:-$(default_chat)}
me=$(hostname -s)
sid=${CLAUDE_CODE_SESSION_ID:-}
[[ $sid =~ ^[A-Za-z0-9._-]{1,128}$ ]] || { echo "chat_post.sh: CLAUDE_CODE_SESSION_ID is not set; not posting" >&2; exit 2; }
body=$(cat)
[ -n "$(printf '%s' "$body" | tr -d '[:space:]')" ] || { echo "chat_post.sh: empty entry; not posting" >&2; exit 2; }
ledger=${XDG_STATE_HOME:-$HOME/.local/state}/kc_chat/$sid.posted
mkdir -p "$(dirname "$ledger")"
header="## $(date '+%Y-%m-%d %H:%M') $me"
fp=$(printf '%s\n' "$body" | python3 -I "$here/chat_entries.py" hash "$header") || exit 1
echo "$fp" >> "$ledger"
printf '\n%s\n%s\n' "$header" "$body" >> "$f" && echo "posted: $header"
