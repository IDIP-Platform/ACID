# Docstring Guide

ACID uses Google-style docstrings for Python code that is part of the public
or reference-facing API. The API reference is generated from Python docstrings
with `mkdocstrings`, so docstrings should be useful to someone reading the
documentation, not only to someone reading the source code.

## Google-Style Baseline

Use triple double quotes:

```python
"""Short summary."""
```

For public functions, classes, and modules, use a docstring when the object is
part of the public API, nontrivial, or has behavior that is not obvious from
its name and signature.

A good docstring should:

- start with a short summary line ending with punctuation
- add a short description paragraph when behavior needs more context
- use `Args:`, `Returns:`, `Yields:`, `Raises:`, `Attributes:`, and `Notes:`
  when relevant
- use consistent hanging indentation
- document `*args` and `**kwargs` with star notation
- describe tuple returns as one tuple return value
- avoid repeating the function name without adding useful information

## ACID Conventions

Prefer type annotations in function signatures. Include types in docstring
entries when the signature is untyped, the type is complex, or the generated
documentation benefits from extra clarity.

When relevant, document:

- array shape expectations
- axis semantics
- dtype behavior
- NaN or infinite-value handling
- label-image assumptions
- accepted string values
- expected file formats
- side effects, such as mutating inputs or writing files

Keep implementation details out of docstrings unless they affect how a caller
should use the function.

## Good Example

```python
def normalize_measurements(
    values: np.ndarray,
    method: str = "zscore",
    axis: int | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Normalize numeric measurements with the selected method.

    NaN values are ignored when estimating normalization statistics and are
    replaced by `fill_value` in the returned array. The returned array has the
    same shape as `values`.

    Args:
        values (np.ndarray): Numeric array containing the measurements to
            normalize.
        method (str): Normalization method to apply. Supported values are
            `"zscore"` and `"minmax"`.
        axis (int | None): Axis used to estimate normalization statistics. If
            `None`, the full array is normalized using global statistics.
        fill_value (float): Value used to replace NaNs in the returned array.

    Returns:
        np.ndarray: Normalized array with the same shape as `values`.

    Raises:
        TypeError: If `values` is not numeric.
        ValueError: If `values` is empty.
        ValueError: If `method` is not supported.

    Notes:
        Use `"zscore"` when values are approximately normally distributed.
        Use `"minmax"` when values should be scaled to the range `[0, 1]`.
    """
```

This example is useful because it describes accepted values, `None` behavior,
NaN handling, shape preservation, return type, and interface-relevant errors.

## Bad Example

```python
def normalize_measurements(values, method="zscore", axis=None, fill_value=0.0):
    """Normalize values.

    Args:
        values: data
        method: method
        axis: axis
        fill_value: fill value

    Returns:
        normalized values
    """
```

This example is weak because it does not explain valid `method` values, what
`axis=None` means, how NaNs are handled, what shape is returned, or what errors
callers should expect.

## Classes and Attributes

Use `Attributes:` for public class attributes that users are expected to read
or configure.

```python
class MeasurementSummary:
    """Summarize measurements from one processing run.

    Attributes:
        name (str): Human-readable name of the processing run.
        n_objects (int): Number of measured objects.
        columns (list[str]): Measurement columns included in the summary.
    """
```

## Generators

Use `Yields:` instead of `Returns:` when a function is a generator.

```python
def iter_batches(values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield fixed-size batches from an array.

    Args:
        values (np.ndarray): Array to split into batches.
        batch_size (int): Maximum number of rows per batch.

    Yields:
        np.ndarray: A batch containing at most `batch_size` rows.
    """
```
