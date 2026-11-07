#!/bin/bash
# SessionStart / SubagentStart hook (.claude/settings.json in this folder) for lab members'
# machines, where the queue has a tree of its own, ~/src/kc_queue (set up by join_queue.sh). It
# makes every shell command of the session use the tree's venv, whatever environment Claude Code
# was started from (it drops an active conda environment's or venv's PATH entries and variables,
# as join_queue.sh does for its build), and tells the session, and each subagent, which checkouts
# and Python are the queue's and that nothing else in ~/src may be changed. On Tom's hosts
# (everything directly in ~/src) the hook command does not run it, and it would do nothing.
input=$(cat)
here=$(cd "$(dirname "$0")" && pwd -P)
tree=$(cd "$here/../../.." && pwd -P)  # scripts/claude_pair_remapping_experiments -> al_analysis -> tree
[ "$(basename "$tree")" = kc_queue ] || exit 0
venv=$tree/al_analysis/venv
event=$(python3 -I -c 'import json, sys; print(json.load(sys.stdin).get("hook_event_name", ""))' \
  <<<"$input" 2>/dev/null)

if [ "$event" = SessionStart ] && [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  path=$(printf '%s\n' "$PATH" | tr ':' '\n' | awk -v c="${CONDA_PREFIX:-}" -v v="${VIRTUAL_ENV:-}" \
      -v q="$venv/bin" '
    $0 == "" || $0 == q { next }
    c != "" && index($0, c) == 1 { next }
    v != "" && index($0, v) == 1 { next }
    /\/(mini|ana)conda[0-9]*(\/|$)|\/mini(forge|mamba)[0-9]*(\/|$)|\/mambaforge(\/|$)|\/\.conda(\/|$)/ { next }
    !seen[$0]++' | paste -sd:)
  unset_vars="PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PIP_USER PIP_PREFIX PIP_TARGET
    CC CXX CPP LD CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD_LIBRARY_PATH $(compgen -e | grep -E '^(CONDA_|_CE_)')"
  {
    echo "unset" $unset_vars
    printf 'export VIRTUAL_ENV=%q PATH=%q\n' "$venv" "$venv/bin${path:+:$path}"
  } >> "$CLAUDE_ENV_FILE"
fi

note="On this machine the queue has a tree of its own, $tree (set up by join_queue.sh): the \
checkouts al_analysis, olfsysm and hong2p in it, and the venv $venv. Use only these, and call the \
venv's Python and pip by their full paths ($venv/bin/python, $venv/bin/pip) or as ./venv/bin/... \
from that al_analysis. Where instructions or notes say ~/src/al_analysis, ~/src/olfsysm or \
~/src/hong2p, they mean the ones in $tree. Never change, build, install from or run commands in \
anything else under ~/src: those checkouts and environments belong to the machine's owner, and a \
guard hook refuses edits and shell commands that name them (reading files there with the Read tool \
is fine). Pass this on to any subagent you start."
python3 -I -c 'import json, sys
print(json.dumps({"hookSpecificOutput": {"hookEventName": sys.argv[1], "additionalContext": sys.argv[2]}}))
' "${event:-SessionStart}" "$note"
