#!/bin/bash
# Sets up this machine to work the KC remapping experiment queue (the scheme in CLAUDE.md in this
# folder), for lab members who want to run queue sessions on their own machines. Safe to re-run:
# it skips what is already done, so re-run it to finish an interrupted setup.
#
# What it does (it changes nothing outside ~/src, ~/.ssh and push_keys/<host>/ in the shared
# directory):
#   1. checks that this checkout is ~/src/al_analysis and that the shared directory is mounted;
#   2. clones olfsysm and hong2p into ~/src (anonymous HTTPS: all three repositories are public);
#   3. builds ./venv with all three installed editable (skipped with --no-install);
#   4. for each repository this machine cannot already push to, makes an SSH deploy key
#      (~/.ssh/kc_queue_<repo>) and sets up that checkout to fetch over HTTPS and push over SSH
#      with that key only (remote.origin.pushurl and core.sshCommand in the checkout's own
#      .git/config, so the machine's other repositories and GitHub logins are untouched);
#   5. copies the public keys to push_keys/<host>/ in the shared directory and prints their
#      fingerprints, to send to Tom (by Slack, email or in person, not through the shared
#      directory) so he can add them with approve_push_keys.sh <host>;
#   6. checks that the experiment inputs load and a model fit runs, and runs the provenance tests
#      (seconds; skipped with --no-tests). --full-tests also runs the whole QUICK=1 test suite,
#      which took 2.5 h on hal.
#
# Usage: ~/src/al_analysis/scripts/claude_pair_remapping_experiments/join_queue.sh [--no-install] [--no-tests | --full-tests]
#        .../join_queue.sh --check    # once Tom has added the keys: can each checkout push?
# PYTHON=<interpreter> picks the Python the venv is made with (default: python3.14).
set -uo pipefail

ORG=ejhonglab
REPOS=(al_analysis olfsysm hong2p)
here=$(cd "$(dirname "$0")" && pwd)
host=$(hostname -s)
src=$HOME/src

die() { echo "join_queue.sh: $*" >&2; exit 1; }
warn() { echo "join_queue.sh: warning: $*" >&2; }
step() { printf '\n== %s\n' "$*"; }

install=1 tests=1 check_only=0
for a in "$@"; do
  case $a in
    --no-install) install=0 ;;
    --no-tests) tests=0 ;;
    --full-tests) tests=2 ;;
    --check) check_only=1 ;;
    -h|--help) sed -n '2,/^set -uo/{/^set -uo/d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
    *) die "unknown option $a (see --help)" ;;
  esac
done
[[ $host =~ ^[A-Za-z0-9._-]{1,64}$ ]] || die "unusable host name '$host'"

al=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || die "$here is not in a git checkout"
[ "$(cd "$src/al_analysis" 2>/dev/null && pwd -P)" = "$(cd "$al" && pwd -P)" ] || die \
  "this checkout is at $al, but the queue's hooks and scripts expect ~/src/al_analysis. Clone it" \
  "there (git clone https://github.com/$ORG/al_analysis ~/src/al_analysis) and run its copy of" \
  "this script."
V=$al/venv/bin

find_shared() {  # the mount points chat_watch.sh and the presence hook look in ($KC_REMAPPING_SHARED: tests)
  local c
  for c in ${KC_REMAPPING_SHARED:-} /local/tensor/Tom/claude_kc_remapping \
      /mnt/tensor/Tom/claude_kc_remapping; do
    timeout 20 ls "$c" >/dev/null 2>&1  # triggers an automount
    [ -f "$c/chat.md" ] && { echo "$c"; return 0; }
  done
  return 1
}

default_branch() {  # $1: checkout. Prints e.g. "main".
  local b
  b=$(git -C "$1" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null) ||
    { git -C "$1" remote set-head origin --auto >/dev/null 2>&1 &&
      b=$(git -C "$1" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null); } || return 1
  echo "${b#origin/}"
}

origin_is_ours() {  # $1: repo name. Is ~/src/<repo>'s origin ejhonglab/<repo> on GitHub?
  local u
  u=$(git -C "$src/$1" remote get-url origin 2>/dev/null) || return 1
  [[ $u =~ github\.com[:/]$ORG/$1(\.git)?/?$ ]]
}

uses_deploy_key() {  # $1: repo name. Was the checkout set up by this script?
  [[ $(git -C "$src/$1" config --get core.sshCommand 2>/dev/null) == *"kc_queue_$1"* ]]
}

try_push() {  # $1: repo name. Dry-run push of the remote's own head: tests write access only.
  local d=$src/$1 b ssh_cmd
  ssh_cmd=$(git -C "$d" config --get core.sshCommand 2>/dev/null || echo ssh)
  # never stop at a password / passphrase prompt
  local noprompt=(GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=never
    "GIT_SSH_COMMAND=$ssh_cmd -o BatchMode=yes")
  b=$(default_branch "$d") || { echo "no default branch for origin"; return 1; }
  env "${noprompt[@]}" timeout 60 git -C "$d" fetch -q origin "$b" 2>&1 || return 1
  env "${noprompt[@]}" timeout 60 git -C "$d" push --dry-run --quiet origin \
    "refs/remotes/origin/$b:refs/heads/$b" 2>&1
}

check_push() {
  local r out ok=0
  for r in "${REPOS[@]}"; do
    if out=$(try_push "$r"); then
      echo "$r: can push"
    else
      ok=1
      echo "$r: cannot push yet:"
      printf '%s\n' "$out" | sed 's/^/    /'
    fi
  done
  return $ok
}

if [ $check_only = 1 ]; then
  check_push && echo "all three checkouts can push." && exit 0
  echo
  echo "If Tom has not added the keys yet (approve_push_keys.sh $host), that is expected."
  exit 1
fi

step "checking this machine"
shared=$(find_shared) || die "the shared directory (Tom/claude_kc_remapping on tensor) is not" \
  "mounted at /local/tensor or /mnt/tensor. Mount tensor at /mnt/tensor (ask whoever manages" \
  "the lab's NAS), or symlink an existing mount there, then re-run this."
[ -w "$shared" ] || die "the shared directory $shared is not writable by $USER"
echo "shared directory: $shared"
missing=()
for t in git ssh ssh-keygen curl python3 timeout; do
  command -v "$t" >/dev/null 2>&1 || missing+=("$t")
done
[ ${#missing[@]} = 0 ] || die "missing commands: ${missing[*]}"
command -v claude >/dev/null 2>&1 || warn "Claude Code (claude) is not installed yet; see" \
  "https://code.claude.com/docs"
if [ -z "$(git config --get user.email)" ] || [ -z "$(git config --get user.name)" ]; then
  warn "git has no user.name / user.email, so commits will fail. Set them with" \
    "git config --global user.name '<name>' and git config --global user.email '<email>'."
fi
prev_info=$shared/push_keys/$host/info.txt
if [ -f "$prev_info" ] && ! grep -qx "user: $USER" "$prev_info"; then
  die "another machine (or user) already joined as '$host' ($prev_info). Host names identify" \
    "machines throughout the queue (chat.md, status files, keys), so this one needs a" \
    "different hostname."
fi

step "checkouts in $src"
for r in olfsysm hong2p; do
  if [ -d "$src/$r/.git" ]; then
    echo "$r: using the existing checkout $src/$r"
  else
    git clone "https://github.com/$ORG/$r.git" "$src/$r" || die "cloning $r failed"
  fi
done
for r in "${REPOS[@]}"; do
  origin_is_ours "$r" || warn "$src/$r: origin is $(git -C "$src/$r" remote get-url origin)," \
    "not $ORG/$r on GitHub; leaving its push setup alone"
done

if [ $install = 1 ]; then
  step "Python environment ($al/venv)"
  if [ ! -x "$V/python" ]; then
    py=${PYTHON:-python3.14}
    command -v "$py" >/dev/null 2>&1 || die "$py not found. On Ubuntu:" \
      "sudo apt install build-essential python3.14-dev python3.14-venv" \
      "(or set PYTHON to another interpreter, 3.8 or newer)"
    "$py" -c 'import ensurepip, sysconfig, os, sys
sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' ||
      die "$py lacks its headers or ensurepip, needed to build olfsysm / make the venv. On" \
        "Ubuntu: sudo apt install build-essential $(basename "$py")-dev $(basename "$py")-venv"
    command -v c++ >/dev/null 2>&1 || die "no C++ compiler (needed to build olfsysm). On" \
      "Ubuntu: sudo apt install build-essential"
    "$py" -m venv "$al/venv" || die "making the venv failed"
  fi
  "$V/python" -m pip install -q -U pip || die "upgrading pip failed"
  # al_analysis pins hong2p and olfsysm to their GitHub versions (exact-requirements.txt), so the
  # editable installs from the checkouts go last, replacing those
  "$V/python" -m pip install -e "$al[dev]" || die "installing al_analysis failed"
  "$V/python" -m pip install -e "$src/hong2p[all]" || die "installing hong2p failed"
  "$V/python" -m pip install -e "$src/olfsysm" || die "building / installing olfsysm failed"
fi

step "push access"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if ! ssh-keygen -F github.com -f "$HOME/.ssh/known_hosts" >/dev/null 2>&1; then
  # GitHub's host keys from its API (over HTTPS), so the first push does not stop at a prompt
  curl -fsS https://api.github.com/meta | python3 -I -c '
import json, sys
for k in json.load(sys.stdin)["ssh_keys"]:
    print("github.com", k)' >> "$HOME/.ssh/known_hosts" || die "could not get GitHub's SSH host keys"
fi
keyed=() own=()
for r in "${REPOS[@]}"; do
  origin_is_ours "$r" || continue
  d=$src/$r key=$HOME/.ssh/kc_queue_$r
  if ! uses_deploy_key "$r" && try_push "$r" >/dev/null; then
    echo "$r: this machine can already push (its own GitHub access); no deploy key needed"
    own+=("$r")
    continue
  fi
  [ -f "$key" ] || ssh-keygen -q -t ed25519 -N '' -C "kc-queue $host $USER $ORG/$r" -f "$key" ||
    die "ssh-keygen failed"
  git -C "$d" remote set-url origin "https://github.com/$ORG/$r.git"
  git -C "$d" remote set-url --push origin "git@github.com:$ORG/$r.git"
  git -C "$d" config core.sshCommand "ssh -i \"$key\" -o IdentitiesOnly=yes -o BatchMode=yes"
  echo "$r: fetches over HTTPS, pushes over SSH with $key"
  keyed+=("$r")
done

dest=$shared/push_keys/$host
mkdir -p "$dest" || die "could not make $dest"
for r in "${REPOS[@]}"; do
  if [[ " ${keyed[*]} " == *" $r "* ]]; then
    cat "$HOME/.ssh/kc_queue_$r.pub" > "$dest/$r.pub" || die "could not write $dest/$r.pub"
  else
    rm -f "$dest/$r.pub"
  fi
done
mem=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
os=$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -sr)
{
  echo "host: $(hostname -f 2>/dev/null || hostname)"
  echo "user: $USER"
  echo "joined: $(date '+%Y-%m-%d %H:%M')"
  echo "cores: $(nproc)"
  echo "memory_gb: ${mem:-?}"
  echo "os: $os"
  echo "python: $( ([ -x "$V/python" ] && "$V/python" -V 2>&1) || echo none)"
  echo "keys_for: ${keyed[*]:--}"
  echo "pushes_with_own_access: ${own[*]:--}"
} > "$dest/info.txt"

if [ $tests -gt 0 ]; then
  step "model and experiment inputs"
  [ -x "$V/python" ] || die "no venv at $al/venv; re-run without --no-install"
  (cd "$here" && "$V/python" - <<'EOF') || die "the model or the experiment inputs do not load (above)"
import kc_remapping_harness as H
from al_analysis import mb_model
problems = mb_model.olfsysm_clean_problems()
if problems:
    raise SystemExit('olfsysm is not a clean build of its checkout (re-run without --no-install): '
        + '; '.join(problems))
print('olfsysm: clean build of its checkout')
print('shared directory:', H.shared_dir())
H.orn16_consensus_pairs()
assert H.yang_data_dir().is_dir(), f'{H.yang_data_dir()} is missing'
print('experiment inputs load')
import os, sys, traceback
# the fit's output (much of it from C++, so redirected at the file-descriptor level) goes to a log
log_path = os.path.abspath('outputs/join_queue_model_check.log')
os.makedirs('outputs', exist_ok=True)
sys.stdout.flush(); sys.stderr.flush()
saved = os.dup(1), os.dup(2)
err = None
with open(log_path, 'w') as log:
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    try:
        resp, _, _, _ = mb_model.fit_mb_model(H.apl16_orn_deltas(), target_sparsity=0.05)
    except Exception:
        err = traceback.format_exc()
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
if err:
    raise SystemExit(f'the model fit failed (its output is in {log_path}):\n{err}')
print(f'model fit runs ({resp.shape[0]} KCs x {resp.shape[1]} odors; output in {log_path})')
EOF
  step "provenance tests"
  (cd "$al" && QUICK=1 "$V/python" -m pytest -q -p no:cacheprovider test/test_model_provenance.py) ||
    die "tests failed (above)"
  if [ $tests = 2 ]; then
    step "QUICK=1 test suite (2.5 h on hal)"
    (cd "$al" && QUICK=1 nice -n 10 "$V/python" -m pytest -q -p no:cacheprovider) ||
      die "tests failed (above)"
  fi
fi

step "done"
if [ ${#keyed[@]} -gt 0 ]; then
  echo "Deploy keys for: ${keyed[*]}. Send Tom these fingerprints (by Slack, email or in person,"
  echo "not through the shared directory):"
  for r in "${keyed[@]}"; do
    echo "    $r: $(ssh-keygen -lf "$HOME/.ssh/kc_queue_$r.pub")"
  done
  echo "and ask him to run approve_push_keys.sh $host. Then check with:"
  echo "    $here/join_queue.sh --check"
else
  echo "This machine can already push to all three repositories; no keys to add."
fi
cat <<EOF

To work the queue, start Claude Code in this folder and accept its workspace-trust prompt (that
enables the hooks in .claude/settings.json here):
    cd $here && claude
and ask it to work the experiment queue. Use Opus 5.5 or newer, or Fable, at xhigh effort or above
for code and write-ups; other models and lower efforts only run scripts that are already written
(CLAUDE.md here, "Permissions by model and effort").
EOF
