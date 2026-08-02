r"""
Check for ``symfn``
"""

# *****************************************************************************
#       Copyright (C) 2026 Mike Hansen <mhansen@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# *****************************************************************************

import os

from . import FeatureTestResult, PythonModule


class Symfn(PythonModule):
    r"""
    A :class:`sage.features.Feature` describing the presence of :ref:`symfn <spkg_symfn>`.

    ``symfn`` is a symmetric-function kernel written in Rust, provided by an
    optional package in the Sage distribution.  When it is present Sage uses it
    in place of Symmetrica for the classical basis conversions and the other
    operations listed in :mod:`sage.libs.symfn.backend`, and it makes the
    operations in :mod:`sage.libs.symfn.extras` available.

    Symmetrica remains the fallback, so nothing in Sage requires this feature.

    Setting the environment variable ``SAGE_DISABLE_SYMFN`` to a nonempty value
    makes the feature absent however the package is installed, which is what
    lets the two backends be compared on identical inputs -- and, because the
    doctest framework asks this same feature, keeps ``# optional - symfn``
    doctests from running against the backend they are not testing.

    EXAMPLES::

        sage: from sage.features.symfn import Symfn
        sage: Symfn().is_present()                        # optional - symfn
        FeatureTestResult('symfn', True)
    """

    def _is_present(self):
        r"""
        Return whether ``symfn`` is importable and not switched off.

        EXAMPLES::

            sage: import os
            sage: from sage.features.symfn import Symfn
            sage: os.environ['SAGE_DISABLE_SYMFN'] = '1'
            sage: bool(Symfn()._is_present())
            False
            sage: del os.environ['SAGE_DISABLE_SYMFN']
        """
        if os.environ.get('SAGE_DISABLE_SYMFN'):
            return FeatureTestResult(self, False,
                                     reason='SAGE_DISABLE_SYMFN is set')
        return super()._is_present()
    def __init__(self):
        r"""
        TESTS::

            sage: from sage.features.symfn import Symfn
            sage: isinstance(Symfn(), Symfn)
            True
        """
        PythonModule.__init__(self, 'symfn', spkg='symfn')


def all_features():
    return [Symfn()]
