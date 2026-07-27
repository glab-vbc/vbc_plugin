# Contributing to vbc_plugin

Thanks for your interest in contributing. This document collects the few
conventions you need to work with the repository. It will grow as the project
adds tests, CI, and linting.

## Development setup

`vbc_plugin` is a HOOMD-blue plugin, built from source against a matching
HOOMD-blue installation. See [`docs/building.md`](docs/building.md) for the full
setup (create the environment, build HOOMD from source, then build the plugin).
In short, with that environment active and the matching C++/CUDA toolchain
loaded, configure and build with the bundled preset.

```bash
cmake --preset default
cmake --build --preset default
cmake --install build
```

## Notebooks, read this before you commit

The example notebooks in `examples/` are committed with their outputs
stripped (code and Markdown only). This keeps the repository small, keeps
`git diff` readable, and prevents stale or machine-specific figures from being
committed. Stripping is enforced by an
[`nbstripout`](https://github.com/kynan/nbstripout) git filter declared in
`.gitattributes`.

The filter's command lives in your local git config (not in the repo), so
after cloning you must install it once.

```bash
pip install nbstripout
nbstripout --install
```

After that, `git add`-ing a notebook automatically strips its outputs from the
commit while leaving your working copy (with figures) untouched. If you skip
this step, git will happily commit the multi-megabyte outputs, so please don't.

To regenerate a notebook's figures, follow the "How to generate the GSD
data" cell at the top of each notebook (it runs `sweep.py`), then run all
cells.

## Documentation maths

The piecewise definitions in [`docs/potential.md`](docs/potential.md) are written
as `math` fenced code blocks, which GitHub typesets directly. VSCode's built-in
Markdown preview shows them as plain text, so to render them locally install a
Markdown maths extension such as Markdown Preview Enhanced or Markdown+Math.

## Commits

Keep commits small and focused, one logical change per commit. Write a
short summary line, then a body that explains the why, not just the what.

## Questions

Open an issue if anything here is unclear or missing.
