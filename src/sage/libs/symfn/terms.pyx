# sage.doctest: needs sage.combinat
# cython: language_level=3, boundscheck=False, wraparound=False
r"""
The per-term loop of the symfn backend

Everything else in :mod:`sage.libs.symfn.backend` runs once per *call*; this
runs once per output *term*, and a conversion of degree 18 produces tens of
thousands of them.  In pure Python that loop measures about 185 ns/term --
unpack a 3-tuple, index a list, build an :class:`Integer`, insert into a dict --
which came to 0.76 times the entire Rust computation it wraps.  None of those
four steps is avoidable in Python, so the loop itself has to stop being Python.

The partitions arrive as **indices** into ``symfn.partitions(degree)`` rather
than as lists of parts, which is what makes the inner step a C array read
instead of building a tuple and hashing it.  Sage's Symmetrica wrapper cannot do
this: :mod:`sage.libs.symmetrica` gets lists of parts back from C and calls
``Partition(res)`` on each, so it pays object construction per term where this
pays an index.

EXAMPLES::

    sage: from sage.libs.symfn.terms import build_terms
    sage: parts = {2: [Partition([1, 1]), Partition([2])]}
    sage: build_terms([(2, 1, 3), (2, 0, -1)], parts)
    {[1, 1]: -1, [2]: 3}

Zero coefficients are dropped, which is what ``_from_dict`` expects::

    sage: build_terms([(2, 1, 0), (2, 0, 5)], parts)
    {[1, 1]: 5}
"""

# ****************************************************************************
#       Copyright (C) 2026 Mike Hansen <mhansen@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

from cpython.dict cimport PyDict_SetItem
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.long cimport PyLong_AsLongAndOverflow
from cpython.tuple cimport PyTuple_GET_ITEM

from sage.rings.integer cimport Integer, smallInteger


cpdef dict build_terms(list raw, dict parts_by_degree):
    r"""
    Turn symfn's indexed output into a dictionary Sage can build an element from.

    INPUT:

    - ``raw`` -- list of triples ``(degree, index, coefficient)``, as returned
      by ``symfn.convert_indexed``; ``index`` is a position in
      ``symfn.partitions(degree)``

    - ``parts_by_degree`` -- dictionary sending a degree to the list of Sage
      :class:`Partition` objects of that degree, in symfn's order

    OUTPUT: a dictionary ``{Partition: Integer}``, with zero coefficients dropped

    EXAMPLES::

        sage: from sage.libs.symfn.terms import build_terms
        sage: build_terms([(1, 0, 7)], {1: [Partition([1])]})
        {[1]: 7}

    Coefficients too wide for a C long are handled by the generic constructor::

        sage: big = 2^70
        sage: build_terms([(1, 0, big)], {1: [Partition([1])]})[Partition([1])] == big
        True
    """
    cdef dict out = {}
    cdef Py_ssize_t k, size = PyList_GET_SIZE(raw)
    cdef object item, degree, index, coeff, table
    cdef Integer value
    cdef long small
    cdef int overflow

    # Nearly every call is homogeneous, so the degree lookup is hoisted and
    # only repeated when the degree actually changes.
    cdef object last_degree = None
    cdef list parts = None

    for k in range(size):
        item = <object>PyList_GET_ITEM(raw, k)
        coeff = <object>PyTuple_GET_ITEM(<tuple>item, 2)
        if not coeff:
            continue
        degree = <object>PyTuple_GET_ITEM(<tuple>item, 0)
        if degree is not last_degree:
            table = parts_by_degree[degree]
            parts = <list>table
            last_degree = degree
        index = <object>PyTuple_GET_ITEM(<tuple>item, 1)
        # ``Integer(coeff)`` parses generically; ``smallInteger`` is the
        # constructor Sage uses internally for values that fit a C long, which
        # every structure constant here does apart from the rare escalated one.
        small = PyLong_AsLongAndOverflow(coeff, &overflow)
        if overflow == 0:
            value = smallInteger(small)
        else:
            value = Integer(coeff)
        PyDict_SetItem(out, <object>PyList_GET_ITEM(parts, <Py_ssize_t>index), value)
    return out
