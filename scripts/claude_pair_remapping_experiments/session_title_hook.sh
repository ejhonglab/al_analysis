#!/bin/bash
# UserPromptSubmit hook that lets a KC-remapping queue session name itself. A prompt that is only
# "__set_session_title__ <label>" (a one-shot CronCreate job the session schedules for itself;
# <label> is the host, e.g. "atlas", or "atlas-2") sets the session title to
# "al_analysis modeling - <label>" and tells Claude to just reply "ok". The prompt is not
# blocked: Claude Code (2.1.319) ignores sessionTitle from a hook that blocks. The title is what
# the Remote Control session list (claude.ai, the phone app) shows and the name other sessions
# address with SendMessage. Every other prompt passes through untouched.
# Registered in ~/.claude/settings.json on each host (see CLAUDE.md in this folder, "Session
# names"); settings are reloaded live, so running sessions pick it up without a restart.
input=$(cat)
case $input in *__set_session_title__*) ;; *) exit 0 ;; esac
exec python3 -I -c '
import json, re, sys
d = json.load(sys.stdin)
m = re.fullmatch(r"\s*__set_session_title__\s+([A-Za-z0-9._-]{1,64})\s*", d.get("prompt") or "")
if m:
    title = "al_analysis modeling - " + m.group(1)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "sessionTitle": title,
            "additionalContext": "This prompt was a session-naming tick, handled by "
                "session_title_hook.sh: the session title is now \"" + title + "\". Nothing "
                "else to do: reply with just \"ok\" and end the turn.",
        },
    }))
' <<<"$input"
