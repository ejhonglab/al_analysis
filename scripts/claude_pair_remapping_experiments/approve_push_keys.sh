#!/bin/bash
# Adds, lists or removes the write-enabled deploy keys that let a lab member's machine push to the
# three repositories the KC remapping queue uses (ejhonglab/al_analysis, olfsysm, hong2p). Tom runs
# it himself, in a terminal, on a host where gh is logged in with admin rights on them; agents run
# it only when Tom asks for it in the terminal, for a host he names (CLAUDE.md here, "Shared
# directory on tensor").
# The machine's owner first runs join_queue.sh there, which writes its public keys to
# push_keys/<host>/ in the shared directory and prints their fingerprints. Compare those with the
# ones this prints before answering y: anyone on the lab network can write to the shared directory.
# A deploy key can push to its one repository only, and is tied to no GitHub account (pushes show
# under the key's title in the audit log, commits under their git author). It does not expire, so
# revoke it when the machine stops working the queue.
#
# Usage: approve_push_keys.sh <host>           # add that host's keys (asks first)
#        approve_push_keys.sh --list           # the keys added so far, and hosts still waiting
#        approve_push_keys.sh --revoke <host>  # remove that host's keys (asks first)
set -uo pipefail

ORG=ejhonglab
REPOS=(al_analysis olfsysm hong2p)
PREFIX=kc-queue

die() { echo "approve_push_keys.sh: $*" >&2; exit 1; }

find_shared() {  # $KC_REMAPPING_SHARED: tests
  local c
  for c in ${KC_REMAPPING_SHARED:-} /local/tensor/Tom/claude_kc_remapping \
      /mnt/tensor/Tom/claude_kc_remapping; do
    timeout 20 ls "$c" >/dev/null 2>&1
    [ -f "$c/chat.md" ] && { echo "$c"; return 0; }
  done
  return 1
}

# this host's keys in one repository: "<id>\t<title>\t<write|read-only>\t<created>" per line
host_keys() {  # $1 repo, $2 host ('' for every key this script added)
  gh api --paginate "repos/$ORG/$1/keys" --jq ".[] | select(.title | startswith(\"$PREFIX $2\"))
    | [(.id|tostring), .title, (if .read_only then \"read-only\" else \"write\" end),
       .created_at] | @tsv"
}

confirm() {  # $1 question. Only an interactive "y" counts.
  local ans=
  read -r -p "$1 [y/N] " ans || true
  [ "$ans" = y ] || [ "$ans" = Y ]
}

command -v gh >/dev/null 2>&1 || die "needs the GitHub CLI (gh), logged in"
gh auth status >/dev/null 2>&1 || die "gh is not logged in (gh auth login)"

valid_host() { [[ $1 =~ ^[A-Za-z0-9._-]{1,64}$ ]] || die "not a host name: '$1'"; }

case ${1:-} in
  --list)
    shared=$(find_shared) || true
    printf '%s\t%s\t%s\t%s\t%s\n' repo id title access created
    approved=
    for r in "${REPOS[@]}"; do
      keys=$(host_keys "$r" "") || die "could not list the deploy keys of $ORG/$r"
      [ -n "$keys" ] && printf '%s\n' "$keys" | sed "s/^/$r\t/"
      approved+=$'\n'$keys
    done
    if [ -n "${shared:-}" ] && [ -d "$shared/push_keys" ]; then
      for d in "$shared"/push_keys/*/; do
        [ -d "$d" ] || continue
        h=$(basename "$d")
        ls "$d"/*.pub >/dev/null 2>&1 || continue
        grep -q $'\t'"$PREFIX $h (" <<<"$approved" ||
          echo "waiting: $h ($(sed -n 's/^user: //p' "$d/info.txt" 2>/dev/null)," \
            "joined $(sed -n 's/^joined: //p' "$d/info.txt" 2>/dev/null))"
      done
    fi
    ;;

  --revoke)
    h=${2:-}; [ -n "$h" ] || die "usage: approve_push_keys.sh --revoke <host>"
    valid_host "$h"
    declare -A ids=()
    for r in "${REPOS[@]}"; do
      keys=$(host_keys "$r" "$h (") || die "could not list the deploy keys of $ORG/$r"
      [ -z "$keys" ] && continue
      printf '%s\n' "$keys" | sed "s/^/$r\t/"
      ids[$r]=$(cut -f1 <<<"$keys" | tr '\n' ' ')
    done
    [ ${#ids[@]} -gt 0 ] || { echo "no keys for $h"; exit 0; }
    confirm "Remove these deploy keys?" || { echo "nothing removed"; exit 1; }
    for r in "${!ids[@]}"; do
      for id in ${ids[$r]}; do
        gh api -X DELETE "repos/$ORG/$r/keys/$id" && echo "$r: removed key $id"
      done
    done
    ;;

  ''|-h|--help)
    sed -n '2,/^set -uo/{/^set -uo/d;s/^# \{0,1\}//;p}' "$0"
    ;;

  -*) die "unknown option $1 (see --help)" ;;

  *)
    h=$1
    valid_host "$h"
    shared=$(find_shared) || die "the shared directory is not mounted"
    dir=$shared/push_keys/$h
    [ -d "$dir" ] || die "$dir does not exist: has $h run join_queue.sh?"
    echo "== $dir/info.txt"
    cat "$dir/info.txt" 2>/dev/null || echo "(missing)"
    user=$(sed -n 's/^user: //p' "$dir/info.txt" 2>/dev/null)
    tmp=$(mktemp -d) || die "mktemp failed"
    trap 'rm -rf "$tmp"' EXIT
    todo=()
    echo
    for r in "${REPOS[@]}"; do
      f=$dir/$r.pub
      if [ ! -f "$f" ]; then
        echo "$r: no key (the machine pushes with its own access, or has no checkout)"
        continue
      fi
      existing=$(host_keys "$r" "$h (") || die "could not list the deploy keys of $ORG/$r"
      if [ -n "$existing" ]; then
        echo "$r: $h already has a key (revoke it first to replace it):"
        printf '%s\n' "$existing" | sed 's/^/    /'
        continue
      fi
      # read the file once, so what is added is exactly what was checked and shown
      key=$(cat "$f")
      [[ $key =~ ^ssh-ed25519\ [A-Za-z0-9+/]+=*(\ [[:print:]]*)?$ ]] ||
        die "$f is not a single ssh-ed25519 public key"
      printf '%s\n' "$key" > "$tmp/$r.pub"
      echo "$r: $(ssh-keygen -lf "$tmp/$r.pub")" || die "ssh-keygen could not read $f"
      todo+=("$r")
    done
    [ ${#todo[@]} -gt 0 ] || { echo "nothing to add"; exit 0; }
    echo
    echo "Check these fingerprints against the ones join_queue.sh printed on $h."
    confirm "Add them as write-enabled deploy keys on $ORG/{${todo[*]}}?" ||
      { echo "nothing added"; exit 1; }
    title="$PREFIX $h (${user:-unknown user}, added $(date +%F))"
    for r in "${todo[@]}"; do
      id=$(gh api -X POST "repos/$ORG/$r/keys" -f title="$title" -f key="$(cat "$tmp/$r.pub")" \
        -F read_only=false --jq .id) || die "adding the key to $ORG/$r failed"
      echo "$r: added key $id ($title)"
    done
    echo
    echo "Done. On $h, join_queue.sh --check should now report that each checkout can push."
    ;;
esac
