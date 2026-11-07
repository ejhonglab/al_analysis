#!/usr/bin/env python3
"""chat.md entries for chat_post.sh / chat_watch.sh: parse, fingerprint, and find new entries
that this session did not write.

An entry is a header line `## YYYY-MM-DD HH:MM <host>` and the lines up to the next header. Its
fingerprint is the sha1 of its header plus its body, with trailing whitespace removed from each
line and leading / trailing blank lines dropped, so the poster and the watcher compute the same
value. A session's ledger (one fingerprint per line, written by chat_post.sh) lists the entries it
posted.

  chat_entries.py hash HEADER < body        print the fingerprint of an entry about to be posted
  chat_entries.py count CHAT_MD             print the number of entries
  chat_entries.py new CHAT_MD START LEDGER  print the entries after the first START that are not in
                                            LEDGER; exit 0 if there are any, 1 otherwise
"""
import hashlib
import re
import sys

HEADER = re.compile(r'^## \d{4}-\d{2}-\d{2} \d{2}:\d{2} \S+\s*$')


def fingerprint(header: str, body_lines) -> str:
    lines = [l.rstrip() for l in body_lines]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha1((header.strip() + '\n' + '\n'.join(lines)).encode()).hexdigest()


def entries(path: str):
    out, cur = [], None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f.read().split('\n'):
            if HEADER.match(line):
                if cur:
                    out.append(cur)
                cur = [line, []]
            elif cur is not None:
                cur[1].append(line)
    if cur:
        out.append(cur)
    return out


def main() -> None:
    cmd = sys.argv[1]
    if cmd == 'hash':
        print(fingerprint(sys.argv[2], sys.stdin.read().split('\n')))
    elif cmd == 'count':
        print(len(entries(sys.argv[2])))
    elif cmd == 'new':
        path, start, ledger = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        try:
            mine = set(open(ledger).read().split())
        except OSError:
            mine = set()
        new = [(h, b) for h, b in entries(path)[start:] if fingerprint(h, b) not in mine]
        for h, b in new:
            print(h)
            print('\n'.join(b).rstrip('\n'))
            print()
        sys.exit(0 if new else 1)
    else:
        sys.exit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
