# Contributing to TurboQuant Pro

Thanks for your interest. Contributions of every size are welcome: a failing test with a
clear report is as useful as a new feature. Merged contributions are credited under your
own GitHub account (see [How PRs are merged](#how-prs-are-merged)).

## Where to start

- Issues labelled [`good first issue`](https://github.com/ahb-sjsu/turboquant-pro/labels/good%20first%20issue)
  are scoped to be done in a few sittings. Each has a triage comment naming the files to
  read first and how the change will be checked.
- Issues labelled [`help wanted`](https://github.com/ahb-sjsu/turboquant-pro/labels/help%20wanted)
  are larger and less mapped out.
- Comment on an issue before starting so it can be assigned to you and nobody duplicates
  the work. If you go quiet for a few weeks, it may be reassigned. Just say so if you are
  still on it.

## Development setup

```bash
git clone https://github.com/ahb-sjsu/turboquant-pro
cd turboquant-pro
pip install -e ".[dev]"
pre-commit install          # optional, runs black and ruff on commit
```

The `[dev]` extra pins the exact formatter and linter versions CI uses
(`black==26.1.0`, `ruff==0.14.8`). GPU packages (cupy, torch) are optional; tests that
need them skip themselves when they are absent.

## Before you open a PR

CI runs these on Python 3.10–3.12; run them locally first:

```bash
ruff check turboquant_pro/ tests/ benchmarks/
black --check turboquant_pro/ tests/ benchmarks/
pytest tests/ -q
```

For the PostgreSQL extension in `pgext/` (Rust, pgrx), CI runs `cargo fmt --check`,
`cargo clippy ... -- -D warnings` and `cargo test` (see `.github/workflows/pgext.yml` for
the exact feature flags).

## What makes a PR easy to merge

- **One change per PR**, with a test that fails without it.
- **Measured claims.** If a PR says something is faster or more accurate, include the
  numbers and how they were measured. For compression quality, the acceptance metric is
  what the consumer sees (recall@k, rank agreement, a task score), never reconstruction
  cosine alone: cosine can read high while rankings collapse.
- **No silent fallbacks.** Prefer an explicit error or a skipped test over code that
  quietly takes a different path (for example, catching `Exception` to hide an import
  failure).
- Keep formatting-only changes out of functional PRs.

## How PRs are merged

Outside contributions are merged with a **merge commit**, so your commits land on
`master` under your name and you appear in the repository's contributor list. If a
maintainer needs to adjust something after merging, it goes in a separate follow-up
commit rather than rewriting yours.

The first time you open a PR, GitHub asks a maintainer to approve the CI run before it
starts; that is a repository safety setting, not a judgement on the PR.

## Reporting bugs

Open an issue with the command you ran, what you expected, what happened (the full
traceback, not a summary), and your environment: OS, Python, `pip show turboquant-pro`,
and GPU/CUDA if relevant.

## License

By contributing you agree that your contribution is licensed under the project's
[MIT License](LICENSE).
