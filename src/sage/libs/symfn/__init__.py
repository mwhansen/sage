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

    EXAMPLES::

        sage: from sage.libs.symfn import is_available
        sage: is_available()                    # optional - symfn
        True
        sage: is_available() in (True, False)
        True
    """
    from sage.features.symfn import Symfn
    return bool(Symfn().is_present())
