#!/bin/bash
# Applies (or, with --check, only shows) the GitHub settings the KC remapping queue relies on, on
# ejhonglab/al_analysis, olfsysm and hong2p, so that a write-enabled deploy key (lab members'
# machines, join_queue.sh) can only add commits to the default branch:
#   - the default Actions token is read-only, and Actions may not approve pull requests (none of
#     the three has workflows of its own; Dependabot and Pages set their own permissions);
#   - ruleset "non-default branches: admins only": creating, updating or deleting any branch but
#     the default one (and dependabot/*) is refused to everyone but repository and organization
#     admins;
#   - ruleset "tags: admins only": the same for tags.
# The existing "no deleting or force push" ruleset on each default branch (nobody exempt) is left
# as it is. Tom runs this, with gh logged in with admin rights on the three repositories; re-running
# it changes nothing that is already in place.
#
# Usage: github_rules.sh [--check]
set -uo pipefail

ORG=ejhonglab
REPOS=(al_analysis olfsysm hong2p)
check=0
case ${1:-} in
  --check) check=1 ;;
  '') ;;
  *) sed -n '2,/^set -uo/{/^set -uo/d;s/^# \{0,1\}//;p}' "$0"; exit 1 ;;
esac

# actor 5 of type RepositoryRole is the repository admin role; OrganizationAdmin covers org owners
BYPASS='[{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"},
  {"actor_id": 1, "actor_type": "OrganizationAdmin", "bypass_mode": "always"}]'
RULES='[{"type": "creation"},
  {"type": "update", "parameters": {"update_allows_fetch_and_merge": false}},
  {"type": "deletion"}]'

ruleset() {  # $1 repo, $2 name, $3 target (branch / tag), $4 JSON list of excluded ref patterns
  local id
  id=$(gh api "repos/$ORG/$1/rulesets" --jq ".[] | select(.name == \"$2\") | .id") || return 1
  if [ -n "$id" ]; then
    gh api "repos/$ORG/$1/rulesets/$id" --jq \
      '"  \(.name): \(.enforcement), excludes \(.conditions.ref_name.exclude), you can bypass: \(.current_user_can_bypass)"'
  elif [ $check = 1 ]; then
    echo "  $2: missing"
  else
    gh api -X POST "repos/$ORG/$1/rulesets" --input - --jq \
      '"  \(.name): created (\(.enforcement)), you can bypass: \(.current_user_can_bypass)"' <<EOF
{"name": "$2", "target": "$3", "enforcement": "active", "bypass_actors": $BYPASS,
 "conditions": {"ref_name": {"include": ["~ALL"], "exclude": $4}}, "rules": $RULES}
EOF
  fi
}

for r in "${REPOS[@]}"; do
  echo "== $ORG/$r"
  if [ $check = 0 ]; then
    gh api -X PUT "repos/$ORG/$r/actions/permissions/workflow" \
      -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false ||
      echo "  setting the Actions token failed"
  fi
  gh api "repos/$ORG/$r/actions/permissions/workflow" --jq \
    '"  Actions token: \(.default_workflow_permissions), may approve pull requests: \(.can_approve_pull_request_reviews)"'
  gh api "repos/$ORG/$r/rulesets" --jq '.[] | "  ruleset: \(.name) (\(.target))"'
  ruleset "$r" "non-default branches: admins only" branch '["~DEFAULT_BRANCH", "refs/heads/dependabot/**"]'
  ruleset "$r" "tags: admins only" tag '[]'
done
