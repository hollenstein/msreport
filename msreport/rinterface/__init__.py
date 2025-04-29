"""Python interface to custome R scripts."""

from msreport.errors import OptionalDependencyError

try:
    from .limma import multi_group_limma, two_group_limma
    from .rinstaller import r_package_version
except ImportError as err:
    raise OptionalDependencyError(
        "R integration is not available. R must be installed and configured before "
        "installing optional R dependencies using 'pip install msreport[R]'. For "
        "more information, see: https://github.com/hollenstein/msreport"
    ) from err


__all__ = ["multi_group_limma", "two_group_limma", "r_package_version"]
