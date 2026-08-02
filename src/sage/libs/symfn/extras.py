# sage.doctest: optional - symfn, needs sage.combinat sage.modules
r"""
Symmetric function operations symfn provides and Sage does not

Everything in :mod:`sage.libs.symfn.backend` computes something Sage already
computes, only faster.  This module is the other half: operations with no
counterpart in Sage, which exist only when the optional
:ref:`symfn <spkg_symfn>` package is installed.

The **Macdonald operator algebra** is the largest piece.  Sage has `\nabla`
(:meth:`~sage.combinat.sf.sfa.SymmetricFunctionAlgebra_generic.Element.nabla`)
and the plethystic `\theta_{q,t}`, but not the eigenoperators the
diagonal-harmonics literature is written in: `\Delta_{e_k}`, `\Delta'_{e_k}`,
`\Theta_{e_k}` and `\Pi`.  Each is diagonal on the modified Macdonald basis
`\tilde{H}_\mu`, and each is what the Delta conjecture and its relatives are
stated with.

EXAMPLES:

`\nabla e_3`, the shuffle theorem's object, in the Schur basis::

    sage: from sage.libs.symfn.extras import nabla_e
    sage: nabla_e(3)
    (q^3+q^2*t+q*t^2+t^3+q*t)*s[1, 1, 1] + (q^2+q*t+t^2+q+t)*s[2, 1] + s[3]

`\Delta'_{e_1} e_2`::

    sage: from sage.libs.symfn.extras import delta_prime_ek
    sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
    sage: s = Sym.schur()
    sage: delta_prime_ek(1, s[2])
    -q*t*s[1, 1]

The Shareshian-Wachs chromatic quasisymmetric function of the path `P_3`::

    sage: from sage.libs.symfn.extras import chromatic_symmetric_function
    sage: chromatic_symmetric_function(graphs.PathGraph(3))                              # needs sage.graphs
    (q^2+4*q+1)*m[1, 1, 1] + q*m[2, 1]

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

import symfn

from sage.combinat.partition import _Partitions
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ


def qt_ring():
    """
    Return `\\QQ(q, t)`, the ring the operators below work over.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import qt_ring
        sage: qt_ring()
        Fraction Field of Multivariate Polynomial Ring in q, t over Rational Field
    """
    return QQ['q', 't'].fraction_field()


def _schur(ring=None):
    """
    Return the Schur basis over ``ring``, defaulting to `\\QQ(q, t)`.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import _schur
        sage: _schur()
        Symmetric Functions over Fraction Field of Multivariate Polynomial Ring in q, t over Rational Field in the Schur basis
    """
    from sage.combinat.sf.sf import SymmetricFunctions
    return SymmetricFunctions(qt_ring() if ring is None else ring).schur()


def _from_qt(rows, ring=None):
    """
    Build a Schur-basis element from symfn's ``(partition, [(a, b, c)])`` rows.

    Each row's second entry lists the monomials of the coefficient: ``(a, b,
    c)`` is `c q^a t^b`.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import _from_qt
        sage: _from_qt([((2,), [(0, 0, 1)]), ((1, 1), [(1, 1, -1)])])
        -q*t*s[1, 1] + s[2]
    """
    s = _schur(ring)
    R = s.base_ring()
    q, t = R('q'), R('t')
    return s._from_dict({_Partitions.from_parts(mu):
                         R.sum(R(c) * q**int(a) * t**int(b) for a, b, c in poly)
                         for mu, poly in rows if poly})


def _to_qt(f):
    """
    Take a Schur-basis element to symfn's ``(partition, [(a, b, c)])`` rows.

    INPUT:

    - ``f`` -- a symmetric function whose coefficients are *polynomials* in `q`
      and `t` with integer coefficients

    The operators below are defined on `\\QQ(q,t) \\otimes \\Lambda` but symfn's
    boundary carries exponent-keyed integer rows, so a genuine denominator or a
    non-integral coefficient is refused here rather than silently truncated.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import _to_qt, _schur
        sage: s = _schur()
        sage: R = s.base_ring(); q, t = R('q'), R('t')
        sage: sorted(_to_qt(s[2] - q*t*s[1, 1]))
        [((1, 1), [(1, 1, -1)]), ((2,), [(0, 0, 1)])]

    A denominator is an error, not a rounding::

        sage: _to_qt(s[2] / (q - t))
        Traceback (most recent call last):
        ...
        ValueError: the coefficient of [2] is not a polynomial in q and t: 1/(q - t)
    """
    s = _schur(f.parent().base_ring())
    rows = []
    for mu, c in s(f).monomial_coefficients().items():
        num = c
        if hasattr(c, 'denominator') and c.denominator() != 1:
            raise ValueError(f"the coefficient of {mu} is not a polynomial in "
                             f"q and t: {c}")
        if hasattr(num, 'numerator'):
            num = num.numerator()
        poly = []
        for exps, coeff in num.monomial_coefficients().items():
            a, b = (int(exps[0]), int(exps[1])) if len(exps) > 1 else (int(exps), 0)
            if coeff not in ZZ:
                raise ValueError(f"the coefficient of {mu} is not integral: {c}")
            poly.append((a, b, int(ZZ(coeff))))
        rows.append((tuple(mu), poly))
    return rows


def nabla_e(n):
    r"""
    Return `\nabla e_n` in the Schur basis.

    This is the object of the shuffle theorem.  It goes through a closed form
    that never divides by an integer, so it is cheaper than
    ``s(e[n]).nabla()``, which changes basis first.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import nabla_e
        sage: nabla_e(2)
        (q+t)*s[1, 1] + s[2]
        sage: nabla_e(3)
        (q^3+q^2*t+q*t^2+t^3+q*t)*s[1, 1, 1] + (q^2+q*t+t^2+q+t)*s[2, 1] + s[3]

    It agrees with Sage's own `\nabla`::

        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s, e = Sym.schur(), Sym.elementary()
        sage: all(nabla_e(n) == s(e[n].nabla()) for n in range(1, 6))
        True
    """
    return _from_qt(symfn.nabla_e(int(n)))


def nabla(f, power=1):
    r"""
    Return `\nabla^{\text{power}} F`.

    The powers share one change of basis rather than paying for it each time,
    which is the difference from iterating Sage's ``nabla``.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import nabla
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s = Sym.schur()
        sage: nabla(s[2])
        -q*t*s[1, 1]
        sage: nabla(s[2], 2) == nabla(nabla(s[2]))
        True

    It agrees with Sage's own ``nabla``, which is the same operator reached
    through the modified Macdonald basis::

        sage: all(nabla(s[mu]) == s(s[mu].nabla())
        ....:     for n in range(1, 6) for mu in Partitions(n))
        True
    """
    rows = _to_qt(f)
    if power == 1:
        return _from_qt(symfn.nabla(rows))
    return _from_qt(symfn.nabla_power(rows, int(power)))


def delta_ek(k, f):
    r"""
    Return `\Delta_{e_k} F`, the operator with eigenvalue `e_k[B_\mu]` on
    `\tilde{H}_\mu`.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import delta_ek
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s = Sym.schur()
        sage: delta_ek(1, s[2])
        -q*t*s[1, 1] + s[2]

    .. NOTE::

        It is `\Delta'_{e_{n-1}}`, not `\Delta_{e_{n-1}}`, that is `\nabla` on
        degree `n` -- see :func:`delta_prime_ek`.  The two differ by whether the
        eigenvalue is `e_k[B_\mu]` or `e_k[B_\mu - 1]`, and on degree 2 that is
        the whole difference between `-qt\,s_{11} + s_2` and `-qt\,s_{11}`.
    """
    return _from_qt(symfn.delta_ek(int(k), _to_qt(f)))


def delta_prime_ek(k, f):
    r"""
    Return `\Delta'_{e_k} F`, the operator with eigenvalue `e_k[B_\mu - 1]` on
    `\tilde{H}_\mu`.

    This is the operator the Delta conjecture is stated with.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import delta_prime_ek
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s = Sym.schur()
        sage: delta_prime_ek(1, s[2])
        -q*t*s[1, 1]

    `\Delta'_{e_{n-1}}` is `\nabla` on degree `n`.  This is the identity that
    distinguishes the primed operator from :func:`delta_ek`, which satisfies no
    such thing::

        sage: from sage.libs.symfn.extras import nabla
        sage: all(delta_prime_ek(n - 1, s[mu]) == nabla(s[mu])
        ....:     for n in range(1, 5) for mu in Partitions(n))
        True

    Its `k = n-1` end on `e_n` is `\nabla e_n`, the shuffle theorem's object::

        sage: from sage.libs.symfn.extras import delta_prime_e, nabla_e
        sage: all(delta_prime_e(n - 1, n) == nabla_e(n) for n in range(1, 6))
        True
    """
    return _from_qt(symfn.delta_prime_ek(int(k), _to_qt(f)))


def delta_prime_e(k, n):
    r"""
    Return `\Delta'_{e_k} e_n` in the Schur basis, the Delta conjecture's object.

    Reaches the answer without expanding `e_n` first.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import delta_prime_e
        sage: delta_prime_e(1, 3)
        (q^2+q*t+t^2+q+t)*s[1, 1, 1] + (q+t+1)*s[2, 1]

    It is the same answer as applying :func:`delta_prime_ek` to `e_n`, only
    without expanding `e_n` first::

        sage: from sage.libs.symfn.extras import delta_prime_ek
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s, e = Sym.schur(), Sym.elementary()
        sage: all(delta_prime_e(k, n) == delta_prime_ek(k, s(e[n]))
        ....:     for n in range(1, 5) for k in range(1, n))
        True
    """
    return _from_qt(symfn.delta_prime_e(int(k), int(n)))


def theta_ek(k, f):
    r"""
    Return `\Theta_{e_k} F`, which raises the degree by `k`.

    Note the cost: `\Theta` expands at degree `n + k`, so it pays for the larger
    degree and not the input's.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import theta_ek
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s = Sym.schur()
        sage: theta_ek(1, s[1]).degree()
        2
    """
    return _from_qt(symfn.theta_ek(int(k), _to_qt(f)))


def big_pi(f):
    r"""
    Return `\Pi F`, the operator with eigenvalue `\Pi_\mu` on `\tilde{H}_\mu`.

    `\Pi^{-1}` is deliberately absent: it is genuinely not a polynomial, so it
    cannot cross symfn's integer boundary.  Only the composite `\Theta` can, and
    :func:`theta_ek` is how to reach it.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import big_pi
        sage: Sym = SymmetricFunctions(QQ['q','t'].fraction_field())
        sage: s = Sym.schur()
        sage: big_pi(s[1])
        s[1]
    """
    return _from_qt(symfn.big_pi(_to_qt(f)))


def qt_kostka(la, mu):
    r"""
    Return the `(q,t)`-Kostka polynomial `K_{\lambda\mu}(q, t)`.

    The convention is `J_\mu = \sum_\lambda K_{\lambda\mu}(q,t)
    S_\lambda(x; t)` -- the integral-form Macdonald polynomial expanded in the
    dual Schur basis, which is the one the positivity theorem is about.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import qt_kostka
        sage: qt_kostka([2, 1], [2, 1])
        q*t + 1
        sage: qt_kostka([3], [1, 1, 1])
        t^3

    It is the convention Sage's own table uses, which is the one thing worth
    checking about a `(q,t)`-Kostka implementation::

        sage: from sage.combinat.sf.macdonald import qt_kostka as sage_qt_kostka
        sage: all(qt_kostka(la, mu) == sage_qt_kostka(la, mu)
        ....:     for n in range(1, 6) for la in Partitions(n) for mu in Partitions(n))
        True

    Off-degree there is no entry rather than a zero one::

        sage: qt_kostka([2], [1, 1, 1])
        Traceback (most recent call last):
        ...
        ValueError: ...
    """
    R = QQ['q', 't']
    q, t = R.gens()
    return R.sum(ZZ(c) * q**int(a) * t**int(b)
                 for a, b, c in symfn.qt_kostka(list(la), list(mu)))


def kronecker_coefficient(la, mu, nu):
    r"""
    Return the single Kronecker coefficient `g^\nu_{\lambda\mu}`.

    Sage reaches these through
    :meth:`~sage.combinat.sf.sfa.SymmetricFunctionAlgebra_generic.Element.itensor`,
    which builds the whole internal product; this answers for one `\nu` without
    doing that.

    EXAMPLES::

        sage: from sage.libs.symfn.extras import kronecker_coefficient
        sage: kronecker_coefficient([2, 1], [2, 1], [2, 1])
        1
        sage: kronecker_coefficient([2, 1], [2, 1], [3])
        1
        sage: kronecker_coefficient([3], [2, 1], [3])
        0

    It agrees with the coefficient read off ``itensor``::

        sage: s = SymmetricFunctions(QQ).schur()
        sage: all(kronecker_coefficient(a, b, c)
        ....:     == s[a].itensor(s[b]).coefficient(c)
        ....:     for a in Partitions(4) for b in Partitions(4) for c in Partitions(4))
        True
    """
    return ZZ(symfn.kronecker_coefficient(list(la), list(mu), list(nu)))


def chromatic_symmetric_function(G):
    r"""
    Return the Shareshian-Wachs chromatic quasisymmetric function `X_\Gamma(x; q)`.

    Computed from the graph's LLT polynomial by the `(q-1)`-plethysm of
    [CM2018]_, in the monomial basis over `\QQ[q]`.  At `q = 1` this is
    Stanley's chromatic symmetric function.

    INPUT:

    - ``G`` -- a graph.  Its vertices must be `0, \ldots, n-1`, and the natural
      orientation `u < v` is the one [CM2018]_ uses.

    .. WARNING::

        Isolated vertices are part of `\Gamma` and are counted.  Dropping them
        is a well-trodden route to a plausible wrong answer, so a graph whose
        vertex set is not `\{0, \ldots, n-1\}` is refused rather than
        relabelled.

    EXAMPLES::

        sage: # needs sage.graphs
        sage: from sage.libs.symfn.extras import chromatic_symmetric_function
        sage: chromatic_symmetric_function(graphs.PathGraph(3))
        (q^2+4*q+1)*m[1, 1, 1] + q*m[2, 1]
        sage: chromatic_symmetric_function(graphs.CompleteGraph(3))
        (q^3+2*q^2+2*q+1)*m[1, 1, 1]

    At `q = 1` it is Stanley's chromatic symmetric function::

        sage: # needs sage.graphs
        sage: G = graphs.PathGraph(4)
        sage: X = chromatic_symmetric_function(G)
        sage: m = SymmetricFunctions(QQ).m()
        sage: m.sum_of_terms((mu, c(q=1)) for mu, c in X) == m(G.chromatic_symmetric_function())
        True
    """
    n = G.num_verts()
    if set(G.vertices()) != set(range(n)):
        raise ValueError(f"the vertices must be 0, ..., {n - 1}, not "
                         f"{sorted(G.vertices())}; isolated vertices count")
    weak = [(int(min(u, v)), int(max(u, v))) for u, v in G.edges(labels=False)]
    rows = symfn.chromatic_from_llt(int(n), weak, [])

    from sage.combinat.sf.sf import SymmetricFunctions
    R = QQ['q']
    q = R.gen()
    m = SymmetricFunctions(R).monomial()
    return m._from_dict({_Partitions.from_parts(mu):
                         R.sum(R(c) * q**int(a) for a, b, c in poly)
                         for mu, poly in rows if poly})
