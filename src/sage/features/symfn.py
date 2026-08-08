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

# The oldest ``symfn`` that answers every call :mod:`sage.libs.symfn.backend`
# and :mod:`sage.libs.symfn.extras` make.  Raise it in the same commit that
# starts using a newer entry point, and raise
# ``build/pkgs/symfn/requirements.txt`` with it.
SYMFN_MINIMUM_VERSION = '0.1.0rc1'


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
        Return whether ``symfn`` is importable, recent enough, and not switched
        off.

        Importability alone is not enough.  A ``symfn`` older than
        ``SYMFN_MINIMUM_VERSION`` imports perfectly well and is missing entry
        points :mod:`sage.libs.symfn.backend` calls, so reporting it present
        would route a conversion into it and raise :exc:`AttributeError` from
        inside the basis machinery instead of falling back to Symmetrica.

        The version is read from ``symfn.__version__`` rather than from the
        distribution metadata, because the two can disagree in exactly the case
        this guards: an old install left behind by a previous ``pip install``
        keeps its ``dist-info`` and reports a version, while the module it
        installed predates the attribute.

        EXAMPLES::

            sage: import os
            sage: from sage.features.symfn import Symfn
            sage: os.environ['SAGE_DISABLE_SYMFN'] = '1'
            sage: bool(Symfn()._is_present())
            False
            sage: del os.environ['SAGE_DISABLE_SYMFN']

        A present feature reports the version it accepted::

            sage: Symfn()._is_present().reason                 # optional - symfn
            'symfn ... is installed'
        """
        if os.environ.get('SAGE_DISABLE_SYMFN'):
            return FeatureTestResult(self, False,
                                     reason='SAGE_DISABLE_SYMFN is set')
        result = super()._is_present()
        if not result:
            return result

        from packaging.version import InvalidVersion, Version

        import symfn
        version = getattr(symfn, '__version__', None)
        if version is None:
            return FeatureTestResult(
                self, False,
                reason=f'symfn is installed at {symfn.__file__} but has no '
                       '__version__, so it predates '
                       f'{SYMFN_MINIMUM_VERSION}; upgrade or remove it')
        try:
            too_old = Version(version) < Version(SYMFN_MINIMUM_VERSION)
        except InvalidVersion:
            return FeatureTestResult(
                self, False,
                reason=f'symfn reports an unreadable version {version!r}')
        if too_old:
            return FeatureTestResult(
                self, False,
                reason=f'symfn {version} is older than the required '
                       f'{SYMFN_MINIMUM_VERSION}')
        return FeatureTestResult(self, True,
                                 reason=f'symfn {version} is installed')

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
