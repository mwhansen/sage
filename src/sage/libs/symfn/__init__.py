r"""
symfn, a symmetric function kernel Sage can use in place of Symmetrica

``symfn`` is an optional package.  When it is installed, the call sites listed
in :mod:`sage.libs.symfn.backend` use it instead of Symmetrica, and the
operations in :mod:`sage.libs.symfn.extras` -- which Symmetrica does not have
-- become available.  When it is not, Symmetrica answers exactly as before, so
nothing in Sage requires it.

Use :func:`is_available` to choose between the two.  Everything that imports
``symfn`` itself lives in :mod:`~sage.libs.symfn.backend` and
:mod:`~sage.libs.symfn.extras`, so importing this module is safe with or
without the package.

Setting the environment variable ``SAGE_DISABLE_SYMFN`` to a nonempty value
makes an installed ``symfn`` invisible, which is how the two backends are
compared on identical inputs without uninstalling anything.
"""

# ****************************************************************************
#       Copyright (C) 2026 Mike Hansen <mhansen@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************


def is_available():
    r"""
    Return whether the optional ``symfn`` package is installed and importable.

    The result is cached by the feature machinery, so this is cheap enough to
    call from a code path that runs often -- but not from one that runs once
    per term.

    Setting ``SAGE_DISABLE_SYMFN`` in the environment answers ``False`` however
    the package is installed.  Every site that chooses a backend consults this
    one function, so that variable puts the whole of Sage back on Symmetrica --
    which is what makes an A/B of the two possible in a process that has the
    package.  The variable is read by :class:`~sage.features.symfn.Symfn`
    rather than here, so the doctest framework sees the same answer and skips
    the ``# optional - symfn`` tests instead of running them against
    Symmetrica.

    EXAMPLES::

        sage: from sage.libs.symfn import is_available
        sage: is_available()                    # optional - symfn
        True
        sage: is_available() in (True, False)
        True

    The switch belongs to :class:`~sage.features.symfn.Symfn`, and has to be
    set before the process starts, because
    :meth:`sage.features.Feature.is_present` caches its answer::

        sage: from sage.features.symfn import Symfn
        sage: import os
        sage: os.environ['SAGE_DISABLE_SYMFN'] = '1'
        sage: bool(Symfn()._is_present())
        False
        sage: del os.environ['SAGE_DISABLE_SYMFN']
    """
    from sage.features.symfn import Symfn
    return bool(Symfn().is_present())
