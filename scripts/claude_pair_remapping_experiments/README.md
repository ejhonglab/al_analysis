# KC remapping experiment queue

Claude Code sessions on several lab machines work through a shared queue of experiments with the
mushroom body model (olfsysm), aimed at reproducing how odor representations change from ORNs to
KCs, with APL on and off. They coordinate through a folder on the lab NAS (tensor) and push their
code to the lab's GitHub repositories. This folder holds the experiment scripts, the queue's rules
(CLAUDE.md) and the tools for adding a machine.

## Joining the queue from your machine

You need Linux (Ubuntu, or another distribution with apt; see below for others), with tensor
mounted at /mnt/tensor or /local/tensor and writable by your account, sudo rights (or someone who
can install packages for you), and a Claude account with Claude Code access. No GitHub account is
needed.

1. Clone the code into its own folder, go to this folder and run the setup script:
   ```
   git clone https://github.com/ejhonglab/al_analysis ~/src/kc_queue/al_analysis
   cd ~/src/kc_queue/al_analysis/scripts/claude_pair_remapping_experiments
   ./join_queue.sh
   ```
   It takes several minutes, and asks before installing anything:
   - the system packages it needs and lacks (Python 3.14 with its headers and venv module, a C++
     compiler, git, ssh, curl). It shows the `sudo apt install` commands and runs them if you
     agree; sudo asks for your password and apt asks once more before installing. On Ubuntu
     releases that do not package Python 3.14 (e.g. 24.04), Python 3.14 comes from the
     deadsnakes PPA, which it adds first (`sudo add-apt-repository ppa:deadsnakes/ppa`, also
     shown and asked about). Python 3.14 is installed next to the system's python3, which stays
     the default;
   - Claude Code, if it is missing (Anthropic's installer, into ~/.local/bin, no sudo);
   - a name and email for your commits, if git has none yet.

   If it stops with a message, fix what it says and run it again; it skips what is already done.
2. At the end it prints the fingerprints of the keys it made, each on a line of its own. Send
   them to Tom by Slack, email or in person (not through the shared folder). It then offers to
   wait until he has added them, checking every minute; if you stop it, `./join_queue.sh
   --check` checks later.
3. Start Claude Code in this folder, accept its workspace-trust prompt, and ask it to work the
   experiment queue (the first time, it also asks you to log in):
   ```
   cd ~/src/kc_queue/al_analysis/scripts/claude_pair_remapping_experiments && claude
   ```
   Use Opus 5.5 or newer, or Fable, at xhigh effort or above, if you want the session to write
   code and write-ups; other models and lower efforts only run scripts that are already written.
4. Ask it to "join the work queue"

Everything the queue uses lives in ~/src/kc_queue: its own copies of al_analysis, olfsysm and
hong2p, and a new Python environment, ~/src/kc_queue/al_analysis/venv. Your other checkouts in
~/src and your other Python environments are never used or changed; hooks remind each session of
this and refuse edits there. Outside ~/src/kc_queue the setup only installs what it asked about
above, adds SSH keys (~/.ssh/kc_queue_*), and writes your machine's public keys and a short
description of it (host, user, cores, memory, OS) to push_keys/<host>/ in the shared folder.

On a distribution without apt, install Python 3.14 (with its headers and venv module), a C++
compiler, git, ssh and curl with its package manager first; the script then finds them.

### Options

`join_queue.sh` options:
- `--no-install`: skip building the Python environment (and installing Python 3.14 and the C++
  compiler for it).
- `--no-tests`: skip the model check at the end. `--full-tests` also runs the whole QUICK=1 test
  suite (2.5 h on hal).
- `--check`: only test whether each checkout can push.
- `PYTHON=<interpreter> ./join_queue.sh`: build the environment from another Python than
  python3.14 (3.8 or newer, not one inside a venv; a conda Python works but gets a warning). The
  script then does not install Python 3.14.

It does the same whether or not a conda environment or venv is active in your shell, and it never
uses or changes a venv it did not make. Machines are identified by their short host name
(`hostname -s`) throughout the queue, so it must differ from every other machine's.

## For Tom: approving a machine and other upkeep

- **Approve its keys.** On one of your hosts, with gh logged in with admin rights on the three
  repositories, run `./approve_push_keys.sh <host>` from this folder in a terminal. Compare the
  fingerprints it prints (each on a line of its own, as join_queue.sh prints them) with the ones
  the machine's owner sent you, since anyone on the lab
  network can write to the shared folder, and answer y. It adds them as write-enabled deploy
  keys, one per repository; join_queue.sh, if the owner left it waiting, then
  reports that each checkout can push (otherwise the owner runs `./join_queue.sh --check`).
- **List and revoke.** `./approve_push_keys.sh --list` shows the keys added so far and the
  machines waiting. Deploy keys do not expire, so run `./approve_push_keys.sh --revoke <host>`
  when a machine stops working the queue. A key can only add ordinary commits to the default
  branch (see the GitHub settings below), but those commits every queue host pulls and runs, so
  sessions on your hosts read pulled changes to the hooks, the setup scripts and the build files
  before running anything (CLAUDE.md, "Lab members' machines").
- **GitHub settings.** `./github_rules.sh` sets what the keys rely on, on all three repositories:
  the default Actions token is read-only, and rulesets refuse creating, updating or deleting any
  branch but the default one (Dependabot's excepted) or any tag to everyone but admins, next to
  the existing ruleset refusing force pushes and deletions of the default branch to everyone.
  Re-running it changes nothing already in place; `./github_rules.sh --check` only shows the
  current settings.
- **Testing on someone else's machine under your own account first.** Don't approve the keys
  from that run, and before they run it under their account, remove push_keys/<host>/ in the
  shared folder (otherwise their run stops, since the host is recorded as yours) and
  ~/.ssh/kc_queue_* in your account there.
- **Republishing pages.** Sessions on lab members' machines publish their artifact pages in their
  own claude.ai accounts, so they never republish the summary or the report page (both in your
  account). They post "For one of Tom's hosts: rebuild and republish the summary" (or the
  report) in chat.md, and a session on one of your hosts does it.
- **Duplicate hooks on your hosts.** The hooks in .claude/settings.json here run the scripts in
  the session's own checkout, so their commands differ from the copies in your dotfiles'
  claude_settings.json. In this folder on hal and atlas each one therefore runs twice, which is
  harmless; giving the dotfiles the same commands would make it run once again.

## Queue monitor

http://hal:17486 shows, on the lab network, which machines take part, what their Claude sessions
are doing, and what is claimed or running. It reads only the shared folder, which the hooks keep
current. The server runs on hal as a systemd user service:
```
systemctl --user link ~/src/al_analysis/scripts/claude_pair_remapping_experiments/monitor-server/claude-queue-monitor.service
systemctl --user enable --now claude-queue-monitor
```
Restart it after changing server.py with `systemctl --user restart claude-queue-monitor`, and read
its logs with `journalctl --user -u claude-queue-monitor`. The one-time steps (letting the service
run without a login, opening the port), the server's options and the meaning of each session
state are in [monitor-server/README.md](monitor-server/README.md).

## Development

The queue's rules (claims, status files, chat, permissions by model and effort, how experiments
are written up) are in [CLAUDE.md](CLAUDE.md). The model code itself is tested through al_analysis
(test/test_mb_model.py, with QUICK=1; see the al_analysis CLAUDE.md).

Tests for the monitor and the provenance checks, from the al_analysis root:
```
./venv/bin/python -m pytest scripts/claude_pair_remapping_experiments/monitor-server
QUICK=1 ./venv/bin/python -m pytest test/test_model_provenance.py
```
