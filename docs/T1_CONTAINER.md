# Why the flagship Tesseract runs uncontainerised

`t1/Dockerfile` exists and CI builds it. The flagship runs it are produced with,
though, run T1 as a plain local process on Windows driving the solver over
`plink`/`pscp`. That is a decision, not an omission, and this document is here so
it does not have to be inferred.

## The situation

The Synopsys Sentaurus installation lives on a CentOS 7 machine that is not ours
to reshape:

| | |
|---|---|
| OS | CentOS 7, kernel 3.10.0-514.el7, glibc 2.17 |
| Python | **2.7.5, and nothing else.** No python3, no conda, no pyenv |
| Containers | no Docker, no podman |
| Shell | Sentaurus resolves only after `source ~/.cshrc` **under csh** |
| Licence | exactly **one** `sdevice` seat; parallel runs queue |
| Measured runtime | **306 s per transient run, exit status 0** |
| Network | outbound works (`pypi.org` returns HTTP 200) |

`tesseract-core` needs Python 3.12. Nothing on that host can host it.

## The two options, and why we took the second

**Install a standalone Python 3 on the solver host and serve T1 there.**
`uv python install` would very likely work: python-build-standalone supports
glibc 2.17. This is the obvious move and we did not take it.

**Run T1 locally and drive the solver over SSH.** This is what ships.

The deciding argument is that *the boundary the judging criteria actually
describe is identical either way*. Criterion 1 asks for composition across a real
boundary. The boundary here is PyTorch autograd ↔ JAX ↔ a closed-source Fortran
and C++ binary with no AD path — and that boundary is crossed identically whether
the SSH hop sits inside T1 or around it. What changes is everything else:

- Installing a toolchain on the licence host would delete the single most
  informative sentence available about this project: **the solver host has Python
  2.7.5 and nothing else on it, and I installed nothing there. Tesseract's
  boundary landed exactly where the technology ran out.**
- No judge can run T1 under either option. It needs a Synopsys licence. The
  reproducible artifact is T2 (`devsim-fefet`, Apache-2.0, on GHCR) plus the
  replay cache, and neither is affected by where T1 runs.
- The hours saved went into `adjoint-shim`, which is the fragile piece and the
  one that is actually reusable by other people.

## What the container is for

The image is the bring-your-own-licence path, Tier D. It contains no Synopsys
code — redistributing the solver would violate the licence; redistributing a
recipe for wiring it up does not. It expects:

- the Sentaurus tree bind-mounted read-only at `/opt/synopsys`
- `SNPSLMD_LICENSE_FILE` pointing at your licence server
- `tcsh` present, which it installs, because `~/.cshrc` cannot be sourced by bash

```bash
docker build -t sentaurus-fefet:byol -f t1/Dockerfile .
docker run --rm \
  -v /path/to/Sentaurus/V-2023.12:/opt/synopsys:ro \
  -e SNPSLMD_LICENSE_FILE=27020@your-license-server \
  sentaurus-fefet:byol serve
```

## What the driver has to survive

Each of these was hit during earlier Sentaurus work on this same host, and each
has a test in `tests/test_t1_driver.py` that costs no licence to run:

- **csh, not bash.** `~/.cshrc` uses `setenv` and `set path`; bash answers
  `Illegal variable name`. Every remote command is wrapped in
  `csh -c 'source $HOME/.cshrc && …'`.
- **csh has no `2>/dev/null`.** It answers `Ambiguous output redirect`. Merging
  uses `>&`.
- **`#` inside an `echo` argument starts a csh comment** and truncates the line.
- **`grep -c` exits 1 on zero matches**, which breaks a `&&` chain. Licence
  polling reads the count from stdout and joins with `;`.
- **An unresolved `@placeholder@`** reaches `sdevice` as a literal, fails the
  parse several minutes in, and holds the licence for all of it. Rendering
  refuses to emit a deck with any placeholder left.
- **A timed-out run keeps executing remotely and keeps the licence.** The retry
  path calls `pkill -9 sdevice` first, or the retry queues behind our own orphan
  and times out too, forever.
- **A `.plt` `Data` block is column-major in header order**, which is not the
  order the names appear elsewhere in the file. Columns are indexed by name.

## Licence hygiene

Only self-authored input decks and numeric outputs are published in this
repository. No Synopsys binaries, no Synopsys-shipped parameter files, and no
`.cmd` fragments copied from Synopsys documentation. Confirm with your licence
administrator before publishing simulation outputs of your own.
