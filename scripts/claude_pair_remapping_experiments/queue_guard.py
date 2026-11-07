"""PreToolUse hook (.claude/settings.json in this folder) for lab members' machines.

There the queue has a tree of its own, ~/src/kc_queue (set up by join_queue.sh), and nothing else
under ~/src (the owner's own checkouts and environments) may be changed. This refuses Edit / Write /
MultiEdit / NotebookEdit calls on files under ~/src outside the tree, and shell commands that name
such a path (absolute, ~/..., $HOME/..., or relative to the command's directory). A shell command
can only be checked by the paths it names, so this catches an agent following instructions written
for the ~/src layout, not a deliberate workaround. On Tom's hosts (everything directly in ~/src)
the hook command does not run it, and it would allow everything.
"""
import json
import os
import re
import shlex
import sys

EDIT_TOOLS = ('Edit', 'Write', 'MultiEdit', 'NotebookEdit')


def tree_and_src():
    """Returns (tree, src): the queue's tree and the folder holding it, or (None, None)."""
    here = os.path.dirname(os.path.realpath(__file__))
    tree = os.path.realpath(os.path.join(here, '..', '..', '..'))
    if os.path.basename(tree) != 'kc_queue':
        return None, None
    return tree, os.path.dirname(tree)


def is_under(path, root):
    return path == root or path.startswith(root + os.sep)


def words(cmd):
    """The shell words of a command, also split at = : , (options, PATH-like lists)."""
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:  # unbalanced quotes
        toks = cmd.split()
    return [w for t in toks for w in re.split(r'[=:,]', t) if w]


def expand(word, home):
    word = re.sub(r'\$\{HOME\}|\$HOME\b', home, word)
    if word == '~' or word.startswith('~/'):
        word = home + word[1:]
    return word


def main():
    tree, src = tree_and_src()
    if tree is None:
        return
    d = json.load(sys.stdin)
    tool = d.get('tool_name') or ''
    ti = d.get('tool_input') or {}
    cwd = d.get('cwd') or os.getcwd()
    home = os.path.expanduser('~')

    def outside(p):
        p = expand(p, home)
        if not os.path.isabs(p):
            p = os.path.join(cwd, p)
        r = os.path.realpath(p)
        return is_under(r, src) and not is_under(r, tree)

    bad = None
    if tool in EDIT_TOOLS:
        p = ti.get('file_path') or ti.get('notebook_path') or ''
        if p and outside(p):
            bad = p
    elif tool == 'Bash':
        for w in words(ti.get('command') or ''):
            if ('/' in w or w.startswith('~') or w.startswith('.')) and outside(w):
                bad = w
                break
    if bad is None:
        return
    reason = (f'{bad} is in {src} outside the queue\'s tree, {tree}. On this machine queue '
        'sessions work only in that tree (its al_analysis, olfsysm and hong2p, and '
        'al_analysis/venv); everything else in ~/src belongs to the machine\'s owner and must not '
        'be changed, built, installed from or run. Use the copy in the tree instead (reading files '
        'with the Read tool is fine).')
    print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny', 'permissionDecisionReason': reason}}))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # never block a tool call because of a bug here
        print(f'queue_guard.py: {e!r}', file=sys.stderr)
        sys.exit(1)
