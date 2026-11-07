#!/bin/bash
# Sets up this machine to work the KC remapping experiment queue (the scheme in CLAUDE.md in this
# folder), for lab members who want to run queue sessions on their own machines. Safe to re-run:
# it skips what is already done, so re-run it to finish an interrupted setup.
#
# What it does (it changes nothing outside ~/src, ~/.ssh and push_keys/<host>/ in the shared
# directory, except git's user.name / user.email in ~/.gitconfig, set only if missing and you agree):
#   1. checks that this checkout is ~/src/al_analysis and that the shared directory is mounted;
#   2. lists the checkouts already in ~/src (commit, date, uncommitted changes, how far behind the
#      lab's default branch). Queue sessions keep them on that branch (they pull, commit, push and
#      reinstall them), so for any that were there before this script (other than an up-to-date
#      al_analysis) it explains this and asks once whether to update them; it exits if you decline,
#      or if one has uncommitted changes, unpushed commits on its default branch or an origin other
#      than the lab's repository. Checkouts it cloned or you agreed to are marked with
#      kcqueue.managed in their .git/config and left alone by later runs. Then it clones olfsysm
#      and hong2p if missing (anonymous HTTPS: all three repositories are public);
#   3. if git has no name or email to commit with, explains why they are needed, asks for them and
#      saves them with git config --global (it exits if you decline: queue sessions commit);
#   4. builds ./venv with all three installed editable (skipped with --no-install);
#   5. for each repository this machine cannot already push to, makes an SSH deploy key
#      (~/.ssh/kc_queue_<repo>) and sets up that checkout to fetch over HTTPS and push over SSH
#      with that key only (remote.origin.pushurl and core.sshCommand in the checkout's own
#      .git/config, so the machine's other repositories and GitHub logins are untouched);
#   6. copies the public keys to push_keys/<host>/ in the shared directory and prints their
#      fingerprints, to send to Tom (by Slack, email or in person, not through the shared
#      directory) so he can add them with approve_push_keys.sh <host>;
#   7. checks that the experiment inputs load and a model fit runs, and runs the provenance tests
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
lab=${KC_QUEUE_LAB_URL:-https://github.com/$ORG}  # the lab's repositories ($KC_QUEUE_LAB_URL: tests)
args=("$@")

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

ident_missing() {  # prints user.name / user.email if some checkout has no value for it
  local k r
  for k in user.name user.email; do
    for r in "${REPOS[@]}"; do
      [ -n "$(git -C "$src/$r" config --get "$k" 2>/dev/null)" ] || { echo "$k"; break; }
    done
  done
}

is_managed() {  # $1: repo name. Was the checkout cloned by this script, or agreed to here?
  [ "$(git -C "$src/$1" config --bool --get kcqueue.managed 2>/dev/null)" = true ]
}

lab_default_branch() {  # $1: repo name. The default branch of the lab's repository, e.g. "main".
  git ls-remote --symref "$lab/$1.git" HEAD 2>/dev/null |
    awk '$1 == "ref:" {sub("^refs/heads/", "", $2); print $2; exit}'
}

plural() {  # $1 count, $2 noun. "1 commit", "2 commits"
  if [ "$1" = 1 ]; then echo "1 $2"; else echo "$1 ${2}s"; fi
}

commit_line() {  # $1 checkout, $2 commit. "<hash> from <date>: <subject>"
  git -C "$1" log -1 --format='%h from %cd: %s' --date=format:'%Y-%m-%d %H:%M' "$2" | cut -c1-90
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
prev_info=$shared/push_keys/$host/info.txt
if [ -f "$prev_info" ] && ! grep -qx "user: $USER" "$prev_info"; then
  die "another machine (or user) already joined as '$host' ($prev_info). Host names identify" \
    "machines throughout the queue (chat.md, status files, keys), so this one needs a" \
    "different hostname."
fi

step "checkouts in $src"
declare -A lab_b=()
problems=() to_update=() to_mark=() to_ask=()
for r in "${REPOS[@]}"; do
  d=$src/$r
  if [ ! -e "$d" ]; then
    echo "$r: not in $src yet; will be cloned from $lab/$r.git"
    continue
  fi
  if [ ! -d "$d/.git" ]; then
    echo "$r: $d is not a git checkout"
    problems+=("$d exists but is not a git checkout")
    continue
  fi
  head_ref=$(git -C "$d" symbolic-ref -q --short HEAD)
  if [ -n "$head_ref" ]; then on="branch $head_ref"; else on="detached HEAD"; fi
  if is_managed "$r"; then
    echo "$r: set up for the queue by an earlier run, left as it is ($on, $(commit_line "$d" HEAD))"
    continue
  fi
  origin=$(git -C "$d" remote get-url origin 2>/dev/null || echo none)
  n_mod=$(git -C "$d" status --porcelain --untracked-files=no | wc -l)
  n_new=$(git -C "$d" ls-files --others --exclude-standard | wc -l)
  if [ "$n_mod" -gt 0 ]; then state="uncommitted changes to $(plural "$n_mod" "tracked file")"; else state=clean; fi
  [ "$n_new" -gt 0 ] && state+=" ($(plural "$n_new" "untracked file"))"
  echo "$r ($d)"
  echo "    origin:   $origin"
  echo "    now on:   $on, $(commit_line "$d" HEAD)"
  echo "    state:    $state"
  if ! origin_is_ours "$r"; then
    problems+=("$r: its origin is $origin, not the lab's repository ($ORG/$r on GitHub)")
    continue
  fi
  b=$(lab_default_branch "$r")
  if [ -z "$b" ] || ! git -C "$d" fetch -q "$lab/$r.git" "+refs/heads/$b:refs/remotes/origin/$b"; then
    problems+=("$r: could not fetch the lab's repository ($lab/$r.git)")
    continue
  fi
  lab_b[$r]=$b
  echo "    lab:      $b, $(commit_line "$d" "origin/$b")"
  [ "$n_mod" -gt 0 ] && problems+=("$r: uncommitted changes to tracked files (git -C $d status)")
  if git -C "$d" show-ref -q --verify "refs/heads/$b"; then
    n_unpushed=$(git -C "$d" rev-list --count "origin/$b..refs/heads/$b")
    [ "$n_unpushed" -gt 0 ] && problems+=("$r: its $b has $(plural "$n_unpushed" commit) not in the lab's $b, which queue sessions would push")
  fi
  if [ -z "$head_ref" ] && [ -z "$(git -C "$d" for-each-ref --count=1 --contains HEAD)" ]; then
    problems+=("$r: its detached HEAD has commits on no branch or tag, which updating would leave behind")
  fi
  n_behind=$(git -C "$d" rev-list --count "HEAD..origin/$b")
  if [ "$head_ref" = "$b" ] && [ "$n_behind" = 0 ]; then
    echo "    update:   none needed (at the lab's $b)"
  elif [ "$head_ref" = "$b" ]; then
    echo "    update:   fast-forward $b by $(plural "$n_behind" commit)"
    to_update+=("$r")
  else
    echo "    update:   switch to $b, at the lab's latest${head_ref:+ (branch $head_ref is kept as it is)}"
    to_update+=("$r")
  fi
  to_mark+=("$r")
  # an up-to-date al_analysis is the clone the setup instructions just made
  [ "$r" = al_analysis ] && [[ " ${to_update[*]} " != *" al_analysis "* ]] || to_ask+=("$r")
done
if [ ${#problems[@]} -gt 0 ]; then
  echo
  echo "These checkouts cannot be used for the queue as they are:"
  printf '  - %s\n' "${problems[@]}"
  die "no checkout was changed. Fix the above, or move those checkouts out of $src (this script" \
    "then clones fresh copies), and re-run this."
fi
if [ ${#to_ask[@]} -gt 0 ]; then
  names=$(printf '%s\n' "${to_ask[@]}" | paste -sd, | sed 's/,\([^,]*\)$/ and \1/; s/,/, /g')
  cat <<EOF

Queue sessions keep these checkouts on the lab's latest default branch: they pull, commit and push,
and reinstall them into ~/src/al_analysis/venv (installing olfsysm rebuilds its compiled module
inside its checkout, replacing one built there by another environment with the same Python
version). Any other environment using them (e.g. with them installed editable) sees every change.
To keep a fixed version for such an environment, answer n, move that checkout out of ~/src and
reinstall it in that environment from its new place, then re-run this: it clones fresh copies.
EOF
  [ -t 0 ] || die "no terminal to ask on; run this in a terminal"
  while :; do
    read -r -p "Update $names as listed above, and let queue sessions keep updating them? (required to continue) [y/N] " ans || ans=n
    case ${ans,,} in
      y|yes) break ;;
      ''|n|no) die "no checkout was changed. Move any checkout another environment relies on" \
        "out of ~/src and re-run this, so it clones fresh copies for the queue." ;;
      *) echo "please answer y or n" ;;
    esac
  done
fi
head_before=$(git -C "$al" rev-parse HEAD)
for r in "${to_update[@]}"; do
  d=$src/$r b=${lab_b[$r]}
  if [ "$(git -C "$d" symbolic-ref -q --short HEAD)" != "$b" ]; then
    if git -C "$d" show-ref -q --verify "refs/heads/$b"; then
      git -C "$d" checkout -q "$b"
    else
      git -C "$d" checkout -q -b "$b" --track "origin/$b"
    fi || die "could not switch $d to $b"
  fi
  git -C "$d" merge -q --ff-only "origin/$b" || die "could not fast-forward $d to the lab's $b"
  echo "$r: now on $b, $(commit_line "$d" HEAD)"
done
for r in "${to_mark[@]}"; do
  git -C "$src/$r" config --bool kcqueue.managed true
done
if [ "$(git -C "$al" rev-parse HEAD)" != "$head_before" ] && [ -x "$here/join_queue.sh" ]; then
  echo "al_analysis was updated, so carrying on with its new copy of this script"
  exec "$here/join_queue.sh" ${args[@]+"${args[@]}"}
fi
for r in olfsysm hong2p; do
  [ -e "$src/$r" ] && continue
  git clone "$lab/$r.git" "$src/$r" || die "cloning $r failed"
  git -C "$src/$r" config --bool kcqueue.managed true
done

step "git name and email"
mapfile -t no_ident < <(ident_missing)
if [ ${#no_ident[@]} -gt 0 ]; then
  what=$(printf '%s\n' "${no_ident[@]}" | sed 's/^user\.//' | paste -sd/ | sed 's|/| and |')
  howto="set them with git config --global user.name '<name>' and git config --global user.email '<email>', then re-run this"
  cat <<EOF
git has no $what set.
Queue sessions commit their code and results and push them to the lab's GitHub repositories, and
git needs a name and an email for every commit: without them it refuses to commit, or makes up an
address from this machine's host name. Both are public on every commit pushed. To have the commits
linked to your GitHub profile, use an email address on your GitHub account (or the noreply address
under GitHub's Settings > Emails). They are saved in ~/.gitconfig, so other repositories on this
account use them too.
EOF
  [ -t 0 ] || die "no terminal to ask on; $howto"
  while :; do
    read -r -p "Set your git $what now? (required to continue) [y/N] " ans || ans=n
    case ${ans,,} in
      y|yes) break ;;
      ''|n|no) die "a git name and email are required; $howto" ;;
      *) echo "please answer y or n" ;;
    esac
  done
  email_re='^[^[:space:]@<>]+@[^[:space:]@<>]+\.[^[:space:]@<>]+$'
  for k in "${no_ident[@]}"; do
    while :; do
      if [ $k = user.name ]; then
        read -r -p "Your name, as shown on commits (e.g. Jane Doe): " v || die "no name entered"
        [ -n "$v" ] && [[ $v != *[\<\>]* ]] && break
        echo "the name cannot be empty or contain < or >"
      else
        read -r -p "Your email (ideally one on your GitHub account): " v || die "no email entered"
        [[ $v =~ $email_re ]] && break
        echo "that does not look like an email address"
      fi
    done
    git config --global "$k" "$v" || die "could not save $k in ~/.gitconfig"
  done
  mapfile -t no_ident < <(ident_missing)
  [ ${#no_ident[@]} = 0 ] || die "some checkout still has no ${no_ident[*]}: its own .git/config" \
    "overrides ~/.gitconfig there"
fi
echo "commits will be by: $(git -C "$al" config user.name) <$(git -C "$al" config user.email)>"

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
