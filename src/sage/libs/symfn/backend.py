# sage.doctest: optional - symfn, needs sage.combinat sage.modules
r"""
symfn as the backend for the classical symmetric function operations

Sage reaches :mod:`sage.libs.symmetrica` from six files.  One of them is the
table of conversions between the five classical bases; the other five call a
Symmetrica function by name.  This module supplies a replacement for each, and
the call sites choose between the two according to
:class:`~sage.features.symfn.Symfn`.

Sage routes every conversion between the classical bases through
:data:`sage.combinat.sf.classical.conversion_functions`, a table of 20 entries
keyed by ``(from_basis, to_basis)``.  That table is the whole integration
surface: replacing its values swaps the backend for all of Sage's
symmetric-function machinery at once, including the paths that only reach a
conversion indirectly -- products in a non-Schur basis, ``scalar``, ``expand``,
plethysm, the Hall-Littlewood and Macdonald bases.

The contract each entry honours, read off Symmetrica's own behaviour:

- the input is a **nonempty** dictionary ``{Partition: coefficient}``; Sage
  guards the empty case itself, and Symmetrica aborts the process rather than
  raising on it;
- coefficients may be rational, and the empty partition is a legal key;
- the return value is anything with ``._monomial_coefficients``.  Symmetrica
  hands back an element over `\ZZ` when every coefficient is integral and over
  `\QQ` otherwise, and this matches that: in the non-`\QQ` path Sage calls
  ``_from_dict`` *without* coercion, so the value types are not merely
  cosmetic.

Denominators are cleared before the call and restored after.  Every conversion
here is linear over the coefficient ring, so scaling by the common denominator
`D` and dividing the result by `D` is exact -- and it is what lets an
integer-only boundary serve a rational input.

EXAMPLES::

    sage: from sage.libs.symfn.backend import conversion_functions
    sage: t = conversion_functions()[('Schur', 'monomial')]
    sage: t({Partition([2, 1]): ZZ(1)})
    2*m[1, 1, 1] + m[2, 1]

AUTHORS:

- Mike Hansen (2026): initial version
"""

# ****************************************************************************
#       Copyright (C) 2026 Mike Hansen <mhansen@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from functools import reduce
from math import lcm

import symfn

from sage.combinat.partition import _Partitions
from sage.libs.symfn.terms import build_terms
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

# Sage's basis names, as they appear as keys in ``conversion_functions``.
NAMES = ['Schur', 'monomial', 'homogeneous', 'elementary', 'powersum']

_TO_SCHUR = {
    'monomial': symfn.monomial_to_schur,
    'homogeneous': symfn.homogeneous_to_schur,
    'elementary': symfn.elementary_to_schur,
    'powersum': symfn.power_to_schur,
}
_FROM_SCHUR = {
    'monomial': symfn.schur_to_monomial,
    'homogeneous': symfn.schur_to_homogeneous,
    'elementary': symfn.schur_to_elementary,
}

# Sage ``Partition`` objects for a whole degree, in symfn's own order.
#
# ``symfn.convert_indexed`` returns the *position* of each output partition in
# ``symfn.partitions(n)`` rather than a list of parts, so the lookup here is a
# list index rather than a hash of a freshly built tuple.  Building that tuple
# and hashing it measured about 40% of everything outside symfn itself.
_PARTS_BY_DEGREE = {}


def _parts(n):
    """
    Return the partitions of ``n`` as Sage objects, in symfn's order.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _parts
        sage: _parts(3)
        [[3], [2, 1], [1, 1, 1]]
    """
    t = _PARTS_BY_DEGREE.get(n)
    if t is None:
        # ``from_parts`` skips validation, which is sound here because symfn
        # enumerates partitions and interns the result for later callers.
        t = _PARTS_BY_DEGREE[n] = [_Partitions.from_parts(p) for p in symfn.partitions(n)]
    return t


def _basis(ring, name):
    """
    Return the classical basis called ``name`` over ``ring``.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _basis
        sage: _basis(ZZ, 'homogeneous')
        Symmetric Functions over Integer Ring in the homogeneous basis
    """
    from sage.combinat.sf.sf import SymmetricFunctions
    sym = SymmetricFunctions(ring)
    return {
        'Schur': sym.schur,
        'monomial': sym.monomial,
        'homogeneous': sym.homogeneous,
        'elementary': sym.elementary,
        'powersum': sym.power,
    }[name]()


def _items(x):
    r"""
    Return the input as ``(partition, coefficient)`` pairs.

    Sage has two calling conventions here.  :mod:`sage.combinat.sf.classical`
    passes a plain ``{Partition: coefficient}`` dictionary, but
    :class:`~sage.combinat.sf.sf.SymmetricaConversionOnBasis` -- the wrapper
    that builds conversion *morphisms*, and therefore the one every non-`\QQ`
    base ring goes through -- passes a
    :class:`~sage.combinat.free_module.CombinatorialFreeModule` element
    instead.

    Only the first convention is visible from the conversion table, so a
    backend that handles only dictionaries appears to work and then fails the
    moment a caller reaches for Macdonald, Jack, Hall-Littlewood, or any base
    ring other than `\QQ`.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _items
        sage: sorted(_items({Partition([2]): 1}))
        [([2], 1)]
        sage: s = SymmetricFunctions(QQ).schur()
        sage: sorted(_items(s[2] + 3*s[1, 1]))
        [([1, 1], 3), ([2], 1)]
    """
    mc = getattr(x, 'monomial_coefficients', None)
    return mc().items() if mc is not None else x.items()


def _convert(d, src, dst):
    """
    Convert ``d``, given in the basis ``src``, to an element in the basis ``dst``.

    The return value is a Sage element, which satisfies both consumers: the
    table path reads ``._monomial_coefficients``, and the morphism path calls
    ``dict(...)``, for which iterating as ``(partition, coefficient)`` pairs is
    exactly right.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _convert
        sage: _convert({Partition([2, 2]): QQ(1)}, 'powersum', 'Schur')
        s[1, 1, 1, 1] - s[2, 1, 1] + 2*s[2, 2] - s[3, 1] + s[4]
        sage: _convert({Partition([3]): ZZ(1)}, 'Schur', 'powersum')
        1/6*p[1, 1, 1] + 1/2*p[2, 1] + 1/3*p[3]
    """
    items = list(_items(d))
    # Clear denominators so the integer boundary can carry the input.
    den = reduce(lcm, (int(QQ(v).denominator()) for _, v in items), 1)
    terms = [(list(k), int(QQ(v) * den)) for k, v in items]

    if src != 'Schur':
        terms = _TO_SCHUR[src](terms)

    # Fast path: an integral input converting to an integral basis, which is
    # nearly every call.  Everything below stays in Python integers and `\ZZ`,
    # and the output is walked **once**.
    #
    # The general path walks it four times -- build with ``QQ(c)/den``, drop
    # zeros, scan the denominators to choose `\ZZ` or `\QQ`, then build the
    # dictionary -- and profiling put that, plus the `\QQ` round trip it
    # implies, above the symfn call itself.  Denominators only ever arise from
    # a rational input or from ``Schur -> powersum``, so the common case should
    # not pay for them.
    if den == 1 and dst != 'powersum':
        raw = symfn.convert_indexed(terms, 'Schur', dst)
        # Degrees come from the *input*: a basis change preserves degree, and
        # the input has a handful of terms where the output has thousands.
        # Deriving them from ``raw`` instead puts a full Python pass back over
        # the output and cancels the compiled loop exactly.
        by_degree = {sum(k): None for k, _ in terms}
        for n in by_degree:
            by_degree[n] = _parts(n)
        return _basis(ZZ, dst)._from_dict(build_terms(raw, by_degree))

    if dst == 'powersum':
        out = [(k, QQ(n) / QQ(dd) / den) for k, (n, dd) in symfn.schur_to_power(terms)]
    elif dst == 'Schur':
        out = [(k, QQ(c) / den) for k, c in terms]
    else:
        out = [(k, QQ(c) / den) for k, c in _FROM_SCHUR[dst](terms)]

    out = [(k, v) for k, v in out if v]
    ring = ZZ if all(QQ(v).denominator() == 1 for _, v in out) else QQ
    target = _basis(ring, dst)
    return target._from_dict({_Partitions.from_parts(k): ring(v) for k, v in out})


def _make(src, dst):
    """
    Return the conversion table entry taking the basis ``src`` to ``dst``.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _make
        sage: _make('monomial', 'Schur').__name__
        't_monomial_Schur_symfn'
    """
    def entry(d):
        return _convert(d, src, dst)

    # Both names, because a table entry is seen through its ``repr`` -- in
    # ``classical.init``'s doctest, and by anyone inspecting the table to find
    # out which backend is installed -- and ``repr`` shows the qualified name.
    entry.__name__ = entry.__qualname__ = f't_{src}_{dst}_symfn'
    return entry


def conversion_functions():
    """
    Return the 20 classical basis conversions, keyed by ``(from, to)``.

    This is what :func:`sage.combinat.sf.classical.init` installs when symfn is
    available.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import conversion_functions
        sage: table = conversion_functions()
        sage: len(table)
        20
        sage: table[('elementary', 'homogeneous')]({Partition([2, 1]): ZZ(1)})
        h[1, 1, 1] - h[2, 1]
    """
    return {(src, dst): _make(src, dst)
            for src in NAMES for dst in NAMES if src != dst}


# --- the five direct call sites --------------------------------------------
#
# Everything below replaces a function Sage calls on
# ``sage.libs.symmetrica.all`` by name, rather than a table entry.  Each
# mirrors its Symmetrica counterpart's return *type*, not just its value: Sage
# consumes these without coercion -- ``sfa._expand`` feeds the result to
# ``resPR(...)``, ``monomial.py`` reads ``.monomial_coefficients()``,
# ``tableau.py`` wraps each element in ``self.element_class``, and
# ``hall_littlewood.py`` calls ``.coefficient(part).subs`` -- so a bare
# dictionary or a list of lists would fail at the caller, not here.


def compute_with_alphabet(basis):
    """
    Return the ``compute_*_with_alphabet`` entry point for ``basis``.

    It expands one basis element as a polynomial in ``n`` indeterminates,
    which is what :meth:`sage.combinat.sf.sfa._expand` hands to ``resPR``.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import compute_with_alphabet
        sage: compute_with_alphabet('Schur')(Partition([2, 1]), 2)
        x0^2*x1 + x0*x1^2
    """
    def entry(part, n, alphabet='x'):
        from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
        n = int(n)
        ring = PolynomialRing(ZZ, n, alphabet)
        terms = symfn.expand_alphabet([(list(part), 1)], basis, n)
        # symfn guarantees distinct exponent vectors and no zero coefficients,
        # and hands each one over as a tuple, so this dictionary needs neither a
        # merging pass nor a key copy.
        return ring({a: ZZ(c) for a, c in terms})

    entry.__name__ = f'compute_{basis}_with_alphabet_symfn'
    return entry


def mult_monomial_monomial(left, right):
    """
    Return the product of two monomial-basis elements.

    Sage only ever calls this with two single-term, coefficient-1 dictionaries
    (see :mod:`sage.combinat.sf.monomial`, which does its own outer double loop
    and its own empty-partition special case -- the latter because Symmetrica
    *aborts the process* on two empty partitions).  Denominators are cleared
    anyway, on the same reasoning as :func:`_convert`: the caller's contract is
    a dictionary of coefficients, and nothing in it promises they are integral.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import mult_monomial_monomial
        sage: mult_monomial_monomial({Partition([2]): 1}, {Partition([1]): 1})
        m[2, 1] + m[3]
    """
    items = ([(list(k), QQ(v)) for k, v in _items(left)],
             [(list(k), QQ(v)) for k, v in _items(right)])
    den = 1
    for side in items:
        for _, v in side:
            den = lcm(den, int(v.denominator()))
    a, b = ([(k, int(v * den)) for k, v in side] for side in items)
    out = [(k, QQ(c) / QQ(den * den)) for k, c in symfn.monomial_multiply(a, b)]
    out = [(k, v) for k, v in out if v]
    ring = ZZ if all(v.denominator() == 1 for _, v in out) else QQ
    return _basis(ring, 'monomial')._from_dict(
        {_Partitions.from_parts(k): ring(v) for k, v in out})


def kostka_number(shape, weight):
    r"""
    Return the Kostka number `K_{\lambda\mu}`, as an :class:`Integer`.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import kostka_number
        sage: kostka_number(Partition([3, 2]), Partition([2, 2, 1]))
        2
    """
    return ZZ(symfn.kostka_number(list(shape), list(weight)))


def kostka_tab(shape, weight):
    """
    Return the semistandard tableaux of shape ``shape`` and weight ``weight``.

    The order is Symmetrica's -- increasing lexicographic in the row-major
    reading word -- and it is part of the interface, not a formatting choice:
    ``SemistandardTableaux(shape, weight).list()`` returns this list verbatim
    and Sage's doctests print it.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import kostka_tab
        sage: kostka_tab(Partition([3, 2]), Partition([2, 2, 1]))
        [[[1, 1, 2], [2, 3]], [[1, 1, 3], [2, 2]]]
    """
    from sage.combinat.tableau import Tableau
    return [Tableau(t) for t in symfn.semistandard_tableaux(list(shape), list(weight))]


def reduced_kronecker_product(la, mu, ring):
    r"""
    Return `\tilde{s}_\lambda \tilde{s}_\mu` as a ``{Partition: coefficient}``
    dictionary over ``ring``.

    The structure constants of the Orellana-Zabrocki irreducible character basis
    are the reduced (stable) Kronecker coefficients, so this is a Kronecker
    computation wearing an ordinary product's clothes.  It is what
    :meth:`sage.combinat.sf.character.IrreducibleCharacterBasis.product_on_basis`
    calls, in place of going through the Schur basis.

    Note that `\tilde{s}_\lambda` is **inhomogeneous** -- it has components in
    every degree from 0 to `|\lambda|` -- so the result's degree is not the sum
    of the inputs'.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import reduced_kronecker_product
        sage: d = reduced_kronecker_product(Partition([2]), Partition([1]), ZZ)
        sage: sorted(d.items())
        [([1], 1), ([1, 1], 1), ([2], 1), ([2, 1], 1), ([3], 1)]
    """
    return {_Partitions.from_parts(nu): ring(c)
            for nu, c in symfn.reduced_kronecker_product(list(la), list(mu))}


def _integral_schur_terms(f):
    r"""
    Return ``f`` in the Schur basis as integer ``(partition, coefficient)``
    pairs, or ``None`` if any coefficient is not an integer.

    symfn's plethysm carries integer rows.  Plethysm preserves the integral
    lattice -- the Schur functions are a `\ZZ`-basis and `f[g]` of two integral
    elements is integral -- so refusing a rational input costs nothing that the
    generic route cannot supply.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _integral_schur_terms
        sage: s = SymmetricFunctions(QQ).s()
        sage: sorted(_integral_schur_terms(s[2] + 3*s[1, 1]))
        [((1, 1), 3), ((2,), 1)]
        sage: _integral_schur_terms(s[2] / 2) is None
        True
    """
    from sage.combinat.sf.sf import SymmetricFunctions
    schur = SymmetricFunctions(f.parent().base_ring()).schur()
    rows = []
    for mu, c in schur(f).monomial_coefficients().items():
        if c not in ZZ:
            return None
        rows.append((tuple(mu), int(ZZ(c))))
    return rows


def plethysm(f, g, parent):
    r"""
    Return the plethysm `f[g]` as an element of ``parent``, or ``None`` if
    either operand has a coefficient that is not an integer.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import plethysm
        sage: Sym = SymmetricFunctions(QQ)
        sage: h, s = Sym.h(), Sym.s()
        sage: plethysm(h[2], h[2], s)
        s[2, 2] + s[4]
        sage: plethysm(s[2] / 2, s[2], s) is None
        True
    """
    a, b = _integral_schur_terms(f), _integral_schur_terms(g)
    if a is None or b is None:
        return None
    R = parent.base_ring()
    schur = parent.realization_of().schur()
    return parent(schur._from_dict({_Partitions.from_parts(mu): R(c)
                                    for mu, c in symfn.plethysm(a, b)}))


def induced_trivial_product(la, mu, ring):
    r"""
    Return `\tilde{h}_\lambda \tilde{h}_\mu` as a ``{Partition: coefficient}``
    dictionary over ``ring``, or ``None`` if the rule declines.

    The Orellana-Zabrocki induced trivial character basis multiplies by a
    double-coset matrix rule, which is cheap exactly where the partitions are
    short and hopeless where they are long: `\tilde{h}_{(6,4)}^2` is a `2 \times
    2` free block of at most 1225 matrices, while `\tilde{h}_{(1^{10})}^2` is a
    `10 \times 10` block with row sums 1, which is `11^{10}`.

    ``None`` is therefore a capacity answer and not a failure, and the caller is
    expected to fall back to the Schur route on it.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import induced_trivial_product
        sage: d = induced_trivial_product(Partition([2]), Partition([1, 1]), ZZ)
        sage: sorted(d.items())
        [([1, 1], 1), ([1, 1, 1], 2), ([2, 1, 1], 1)]

    A long pair declines rather than enumerating::

        sage: induced_trivial_product(Partition([1] * 10), Partition([1] * 10), ZZ) is None
        True
    """
    rows = symfn.ht_multiply([(tuple(la), 1)], [(tuple(mu), 1)])
    if rows is None:
        return None
    return {_Partitions.from_parts(nu): ring(c) for nu, c in rows}


# --- Schubert polynomials ---------------------------------------------------
#
# These four are the sites in :mod:`sage.combinat.schubert_polynomial` that
# reach Symmetrica with no fallback of their own.  ``divided_difference`` is
# deliberately not here: its default is ``algorithm='sage'``, a pure-Python
# implementation, and its ``algorithm='symmetrica'`` branch names the backend it
# wants -- answering that with a different one would make the argument a lie.
#
# ``scalar_product`` is not here either, and will not be: it needs
# ``scalarproduct_schubert``, the one operation symfn does not have, and nothing
# in sagelib calls it (see the coverage audit).  It stays with the optional
# Symmetrica package.


def _schub_terms(elt):
    """
    Return a Schubert element as symfn's ``(one-line word, coefficient)`` pairs.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _schub_terms
        sage: X = SchubertPolynomialRing(ZZ)
        sage: sorted(_schub_terms(X([3, 2, 1]) + 2*X([2, 1])))
        [([2, 1], 2), ([3, 2, 1], 1)]
    """
    return [(list(w), int(c)) for w, c in elt.monomial_coefficients().items()]


def _schub_from(rows, parent):
    """
    Build an element of ``parent`` from symfn's ``(one-line word, coefficient)``
    pairs.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import _schub_from
        sage: X = SchubertPolynomialRing(ZZ)
        sage: _schub_from([((2, 1), 3)], X)
        3*X[2, 1]
    """
    from sage.combinat.permutation import Permutation
    R = parent.base_ring()
    return parent._from_dict({Permutation(list(w)).remove_extra_fixed_points(): R(c)
                              for w, c in rows if c})


def schubert_multiply(left, right, parent):
    """
    Return the product of two Schubert basis elements.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import schubert_multiply
        sage: X = SchubertPolynomialRing(QQ)
        sage: schubert_multiply(Permutation([3,2,1]), Permutation([2,1,3]), X)
        X[4, 2, 1, 3]
    """
    return _schub_from(symfn.schubert_multiply([(list(left), 1)],
                                               [(list(right), 1)]), parent)


def schubert_multiply_variable(elt, i):
    """
    Return `x_i \\cdot f`, the signed Monk rule.

    ``i`` is **0-based**, which is Sage's convention here and Symmetrica's;
    symfn's own entry point is 1-based, and this is where the two are
    reconciled.  Symmetrica is 0-based for this operation and 1-based for
    ``divdiff_schubert``, so the adapter cannot pick one rule and apply it
    everywhere.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import schubert_multiply_variable
        sage: X = SchubertPolynomialRing(ZZ)
        sage: schubert_multiply_variable(X([3, 2, 4, 1]), 0)
        X[4, 2, 3, 1]
        sage: schubert_multiply_variable(X([3, 2, 4, 1]), 2)
        X[3, 2, 5, 1, 4] - X[3, 4, 2, 1] - X[4, 2, 3, 1]
    """
    rows = symfn.schubert_multiply_variable(_schub_terms(elt), int(i) + 1)
    return _schub_from(rows, elt.parent())


def schubert_expand(elt, ring):
    """
    Return a Schubert polynomial expanded into monomials, over ``ring``.

    The number of variables is the length of the longest permutation in the
    support -- not the width the monomials happen to need -- because that is
    what Symmetrica does and the parent is visible: ``X([1,3,2]).expand()`` is
    ``x0 + x1`` in a **three**-variable ring.  The empty permutation is the
    degenerate case, and gets one variable rather than none.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import schubert_expand
        sage: X = SchubertPolynomialRing(ZZ)
        sage: schubert_expand(X([1, 3, 2]), ZZ)
        x0 + x1
        sage: schubert_expand(X([1, 3, 2]), ZZ).parent()
        Multivariate Polynomial Ring in x0, x1, x2 over Integer Ring
        sage: schubert_expand(X([1, 2]), ZZ).parent()
        Multivariate Polynomial Ring in x0 over Integer Ring
    """
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    rows = _schub_terms(elt)
    n = max([len(w) for w, _ in rows] + [1])
    terms = symfn.schubert_expand(rows)
    R = PolynomialRing(ring, n, [f'x{i}' for i in range(n)])
    return R({tuple(a) + (0,) * (n - len(a)): ring(c) for a, c in terms})


def polynomial_to_schubert(poly, parent):
    """
    Return a polynomial written in the Schubert basis.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import polynomial_to_schubert
        sage: X = SchubertPolynomialRing(ZZ)
        sage: R.<x0, x1> = PolynomialRing(ZZ)
        sage: polynomial_to_schubert(x0 + x1, X)
        X[1, 3, 2]
    """
    terms = [(list(e), int(c)) for e, c in poly.monomial_coefficients().items()]
    return _schub_from(symfn.polynomial_to_schubert(terms), parent)


def hall_littlewood(part):
    r"""
    Return `Q'_{\text{part}}` in the Schur basis, over `\ZZ[x]`.

    The parameter is called ``x`` and not ``t`` because that is the ring
    Symmetrica hands back and :mod:`sage.combinat.sf.hall_littlewood`
    substitutes for it by name; returning `\ZZ[t]` would raise there rather
    than differ quietly.

    EXAMPLES::

        sage: from sage.libs.symfn.backend import hall_littlewood
        sage: hall_littlewood(Partition([2, 1]))
        s[2, 1] + x*s[3]
    """
    ring = ZZ['x']
    rows = symfn.hall_littlewood(list(part))
    return _basis(ring, 'Schur')._from_dict(
        {_Partitions.from_parts(mu): ring({int(e): ZZ(c) for e, c in poly})
         for mu, poly in rows})
