#!/usr/bin/env python3
"""Finds the names other code imports from al_analysis's main modules.

Scans (statically, with `ast`) the Python files of al_analysis's scripts/, test/,
test_scripts/ and top-level scripts, the al_analysis package itself (its modules import
from each other), and the experiment queue repository (model_experiment_queue, next to
this checkout), for every name taken from `al_analysis.mb_model`,
`al_analysis.al_util` and `al_analysis.al_analysis` (`from ... import x`, or `mod.x` on
an imported module), and every module attribute those files assign (`al_util.verbose =
True`), which only works if the name lives in that module itself.

Run as a script to regenerate the frozen list `test/test_data/import_surface.json`, which
`test_import_surface.py` checks is still importable:
```
./venv/bin/python test/import_surface.py [--queue-root PATH]
```
Dynamic imports (`importlib.import_module`, `getattr(mod, name)`) are not seen.
"""

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple
import warnings


test_dir: Path = Path(__file__).resolve().parent
repo_root: Path = test_dir.parent
frozen_path: Path = test_dir / 'test_data' / 'import_surface.json'

MODULES: Tuple[str, ...] = ('al_analysis.mb_model', 'al_analysis.al_util',
    'al_analysis.al_analysis'
)

default_queue_root: Path = repo_root.parent / 'model_experiment_queue'

_skip_dir_names = {'venv', '.venv', 'build', 'site-packages', '__pycache__', '.git',
    'outputs', '.tox', 'node_modules'
}


def _git_py_files(root: Path, pathspecs: Iterable[str]) -> Optional[List[Path]]:
    """Returns tracked and untracked-but-not-ignored .py files, or None if not in git.
    """
    try:
        out = subprocess.run(['git', '-C', str(root), 'ls-files', '--cached', '--others',
                '--exclude-standard', '--', *pathspecs], capture_output=True, text=True,
            check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    files = []
    for line in out.splitlines():
        if not line.endswith('.py'):
            continue
        p = root / line
        if p.is_file():
            files.append(p)
    return files


def _walk_py_files(root: Path) -> List[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _skip_dir_names]
        files.extend(Path(dirpath) / f for f in filenames if f.endswith('.py'))
    return files


def source_files(queue_root: Optional[Path] = None) -> Dict[str, List[Path]]:
    """Returns dict of source label -> Python files to scan.
    """
    if queue_root is None:
        queue_root = default_queue_root

    al_specs = {
        'al_analysis/scripts': ['scripts/*.py', 'scripts/**/*.py'],
        'al_analysis/test': ['test/*.py', 'test/**/*.py'],
        'al_analysis/test_scripts': ['test_scripts/*.py'],
        'al_analysis/top-level': [':(glob)*.py'],
        'al_analysis/package': ['al_analysis/*.py'],
    }
    sources = dict()
    for label, specs in al_specs.items():
        files = _git_py_files(repo_root, specs)
        if files is None:
            subdir = label.split('/', 1)[1]
            if subdir == 'top-level':
                files = sorted(repo_root.glob('*.py'))
            elif subdir == 'package':
                files = sorted((repo_root / 'al_analysis').glob('*.py'))
            else:
                files = _walk_py_files(repo_root / subdir)
        sources[label] = sorted(set(files))

    if queue_root.is_dir():
        files = _git_py_files(queue_root, ['*.py'])
        if files is None:
            files = _walk_py_files(queue_root)
        sources['model_experiment_queue'] = sorted(set(files))

    return sources


def _module_of_package_attr(name: str) -> Optional[str]:
    full = f'al_analysis.{name}'
    return full if full in MODULES else None


class _Visitor(ast.NodeVisitor):
    def __init__(self, own_module: Optional[str] = None):
        # local name -> module (aliases of the scanned modules)
        self.aliases: Dict[str, str] = dict()
        # module -> names imported / read as attributes
        self.used: Dict[str, Set[str]] = defaultdict(set)
        # module -> attributes assigned from outside the module
        self.assigned: Dict[str, Set[str]] = defaultdict(set)
        self.own_module = own_module

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in MODULES:
                if alias.asname is not None:
                    self.aliases[alias.asname] = alias.name
                else:
                    # `import al_analysis.mb_model` binds `al_analysis`. accessed as
                    # attribute chain, handled in `_attr_chain_module`
                    self.aliases.setdefault('al_analysis', 'al_analysis')

            elif alias.name == 'al_analysis':
                self.aliases[alias.asname or 'al_analysis'] = 'al_analysis'

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ''
        if node.level and self.own_module is not None:
            # relative import within the al_analysis package
            module = f'al_analysis.{module}' if module else 'al_analysis'

        if module in MODULES:
            for alias in node.names:
                if module == self.own_module:
                    continue
                self.used[module].add(alias.name)

        elif module == 'al_analysis':
            for alias in node.names:
                full = _module_of_package_attr(alias.name)
                if full is not None:
                    self.aliases[alias.asname or alias.name] = full

        self.generic_visit(node)

    def _attr_module(self, node: ast.Attribute) -> Optional[str]:
        """Returns scanned module that `node.value` refers to, if any."""
        value = node.value
        if isinstance(value, ast.Name):
            module = self.aliases.get(value.id)
            if module in MODULES:
                return module
            return None

        # al_analysis.mb_model.<x>
        if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and
                self.aliases.get(value.value.id) == 'al_analysis'):
            return _module_of_package_attr(value.attr)

        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        module = self._attr_module(node)
        if module is not None and module != self.own_module:
            if isinstance(node.ctx, ast.Store):
                self.assigned[module].add(node.attr)
            self.used[module].add(node.attr)

        self.generic_visit(node)


    def visit_Call(self, node: ast.Call) -> None:
        # setattr(mod, 'x', ...), monkeypatch.setattr(mod, 'x', ...), getattr(mod, 'x')
        func = node.func
        fn_name = None
        if isinstance(func, ast.Name):
            fn_name = func.id
        elif isinstance(func, ast.Attribute):
            fn_name = func.attr

        # (getattr w/ a default and monkeypatch.setattr(..., raising=False) are for
        # attributes that may not exist)
        optional = len(node.args) >= 3 and fn_name == 'getattr' or any(
            kw.arg == 'raising' and isinstance(kw.value, ast.Constant) and
            kw.value.value is False for kw in node.keywords
        )
        if (not optional and fn_name in ('setattr', 'getattr', 'hasattr', 'delattr') and
                len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and
                isinstance(node.args[1], ast.Constant) and
                isinstance(node.args[1].value, str)):

            module = self.aliases.get(node.args[0].id)
            if module in MODULES and module != self.own_module:
                name = node.args[1].value
                self.used[module].add(name)
                if fn_name in ('setattr', 'delattr'):
                    self.assigned[module].add(name)

        self.generic_visit(node)


def scan_file(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Returns (module -> names used, module -> names assigned) for one file.
    """
    try:
        with warnings.catch_warnings():
            # e.g. invalid escape sequences in old scripts
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'),
                filename=str(path)
            )
    except SyntaxError as err:
        print(f'skipping {path} (could not parse: {err})', file=sys.stderr)
        return dict(), dict()

    own_module = None
    try:
        rel = path.resolve().relative_to(repo_root / 'al_analysis')
        if len(rel.parts) == 1:
            own_module = f'al_analysis.{rel.stem}'
    except ValueError:
        pass

    visitor = _Visitor(own_module=own_module)
    visitor.visit(tree)
    return visitor.used, visitor.assigned


def scan(queue_root: Optional[Path] = None) -> Dict:
    """Returns the import surface, as frozen in `frozen_path`.

    Dict with:
    - 'modules': {module: {name: [source labels using it]}}
    - 'assigned': {module: {name: [source labels assigning it]}}
    - 'sources': {source label: number of files scanned}
    """
    modules: Dict[str, Dict[str, Set[str]]] = {m: defaultdict(set) for m in MODULES}
    assigned: Dict[str, Dict[str, Set[str]]] = {m: defaultdict(set) for m in MODULES}
    n_files = dict()
    for label, files in source_files(queue_root).items():
        n_files[label] = len(files)
        for path in files:
            used, assigned_here = scan_file(path)
            for module, names in used.items():
                for name in names:
                    modules[module][name].add(label)

            for module, names in assigned_here.items():
                for name in names:
                    assigned[module][name].add(label)

    def to_plain(d):
        return {m: {n: sorted(labels) for n, labels in sorted(names.items())}
            for m, names in d.items()
        }

    return dict(modules=to_plain(modules), assigned=to_plain(assigned),
        sources=n_files
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--queue-root', type=Path, default=default_queue_root,
        help=f'experiment queue checkout to also scan (default: {default_queue_root})'
    )
    parser.add_argument('-o', '--output', type=Path, default=frozen_path,
        help=f'where to write the JSON (default: {frozen_path})'
    )
    args = parser.parse_args()

    if not args.queue_root.is_dir():
        print(f'warning: {args.queue_root} not found. its imports will be missing from '
            'the frozen list', file=sys.stderr
        )

    surface = scan(args.queue_root)
    args.output.write_text(json.dumps(surface, indent=1, sort_keys=True) + '\n')
    n_names = {m: len(names) for m, names in surface['modules'].items()}
    print(f'wrote {args.output}: {n_names} names; files scanned: {surface["sources"]}')


if __name__ == '__main__':
    main()
