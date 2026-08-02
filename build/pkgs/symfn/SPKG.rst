symfn: A symmetric function kernel in Rust
==========================================

Description
-----------

symfn computes with symmetric functions: the six classical bases and the
conversions between them, Kostka numbers and tableaux, characters of the
symmetric group, plethysm, the Kronecker product, and the Hall-Littlewood,
Jack, Macdonald and LLT families.

Sage uses it in two ways, and neither is required -- without the package Sage
behaves exactly as it did before:

* as a **backend**, in place of Symmetrica, for the conversions between the
  classical bases and for the five operations listed in
  ``sage.libs.symfn.backend``.  The answers are the same; ``symfn`` is faster
  on the paths that were slow.

* as **new capability**, through ``sage.libs.symfn.extras``, for operations
  Sage has no implementation of -- the `\Delta_f`, `\Delta'_f` and `\Theta_f`
  operators of the Macdonald operator algebra among them.

The Python surface is plain data -- lists of ``(partition, coefficient)``
pairs, partitions as tuples of parts, coefficients as Python integers of any
size -- and contains no reference to Sage.  Everything Sage-shaped lives on
this side of the boundary, in ``sage.libs.symfn``.

License
-------

MIT License or Apache License 2.0, at your option

Upstream Contact
----------------

- PyPI: https://pypi.org/project/symfn/

Dependencies
------------

Python (>= 3.9).  Build dependencies: pip, packaging.

Special Notes
-------------

symfn is distributed as a binary wheel and Sage never compiles it: the
extension is built against Python's stable ABI (``abi3-py39``), so one wheel
per platform covers every supported Python version rather than one per
(version, platform) pair.  A source build needs a Rust toolchain, which is a
packager concern rather than an end-user one -- as with ``rpds_py``, the other
maturin-built Rust package in the Sage distribution.

The compiled part of the Sage-side adapter -- the per-term marshalling loop in
``sage.libs.symfn.terms`` -- is built with Sage rather than shipped in the
wheel, because it ``cimport``\ s Sage's ``Integer`` and so has to match a
specific Sage build.  This is the same arrangement ``lrcalc`` uses: an
independently released library, wrapped by thin code that lives in and is built
with Sage.
