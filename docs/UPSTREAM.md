# Upstream contributions

Two, both found by using the tools rather than by reading them, and both with the
motivating case sitting in this repository.

---

## 1. `tesseract_jax.apply_tesseract` crashes on any non-array output leaf

**Status:** found D1, worked around here, PR to open.
**Severity:** `TypeError` with a message that points nowhere near the cause.
**Fix size:** one line.

### What happens

A Tesseract whose `OutputSchema` contains a `str` or `bool` field cannot be used
with `apply_tesseract` at all:

```
TypeError: string indices must be integers, not 'str'
```

raised from `tesseract_jax/primitive.py`, inside `apply_tesseract`.

### Why

`abstract_eval` returns a pytree that mixes shape/dtype dicts with plain values
for the non-array fields. The code checks for that case and then does not act on
it:

```python
is_aval = lambda x: isinstance(x, dict) and "dtype" in x and "shape" in x
flat_avals, output_pytreedef = jax.tree.flatten(avals, is_leaf=is_aval)
for aval in flat_avals:
    if not is_aval(aval):
        continue            # <- non-aval leaves are tolerated here ...
    _check_dtype(aval["dtype"])

flat_avals = tuple(
    jax.ShapeDtypeStruct(shape=tuple(aval["shape"]), dtype=aval["dtype"])
    for aval in flat_avals   # ... and unconditionally subscripted here
)
```

The `continue` shows the intent. The comprehension immediately below it does not
honour it.

### Suggested fix

Carry non-aval leaves through as static output structure rather than converting
them, or fail early with a message naming the offending field, e.g.

> `OutputSchema field 'backend' has type str; every output leaf of a Tesseract
> used with apply_tesseract must be an array.`

Either is a large improvement on `string indices must be integers`.

### Motivating case

This project wants `backend` and `content_hash` on every oracle output: they are
the evidence that a forward value came from the solver it is attributed to. Both
had to be moved out of the schema and into a side-channel
(`results/runs/provenance.jsonl` and the cache record). That is arguably a better
design for an audit trail, but it was forced rather than chosen.

---

## 2. `eps` is a scalar on all three finite-difference helpers

**Status:** identified in the build spec, confirmed D1 by running into it.
**Fix size:** small, plus tests.

`tesseract_core/runtime/experimental/finite_differences.py` takes `eps` as a
scalar `float` on all three functions. The `check-gradients` docstring already
warns that the value has to be chosen against the inputs:

> Finite difference approximations are sensitive to numerical precision. When
> finite differences are reported incorrectly as 0.0, it is likely that the
> chosen `eps` is too small, especially for inputs that do not use float64
> precision.

It warns about too small. This repository hits the same argument from the other
end, where a step larger than the value it perturbs replaces that value instead.

### The confirmation

`tesseract-runtime check-gradients` inherits the same scalar. Running V3 against
`snn-lif-ecg` shows exactly what the docs warn about: the five inputs are

| input | nominal |
|---|---|
| `beta` | 0.60 |
| `th_th` | 5.0 |
| `sig_w` | 0.23 |
| `g_min` | 2.6e-5 |
| `g_max` | 2.0e-4 |

No single `eps` is right for both `th_th` at 5.0 and `g_min` at 2.6e-5. An `eps`
of 1e-4 is four times `g_min` itself — not a perturbation of it but a replacement
of it — while the same step is negligible against `th_th`.

This project's device parameters have the same problem in the raw: `t_fe` in
nanometres alongside `N_ch` in cm^-3, eighteen orders apart. The frozen contract
sidesteps it by normalising `theta` to `[0,1]^D`, which is a real design decision
and also the reason a single scalar `alpha` works in `shim/adjoint.py` — but not
every caller can normalise, and the helper should not require it.

### Proposal

Accept a vector or per-leaf-pytree `eps`, keeping the scalar path as the default
so nothing breaks. Motivating case: this repository, where the alternative was to
normalise the entire design space before it could be differenced at all.
