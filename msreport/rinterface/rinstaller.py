import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


def r_package_version(package_name: str) -> (str, str):
    """Returns the version number of an installed R package."""
    with robjects.conversion.localconverter(robjects.default_converter):
        utils = importr("utils")
        version_print = utils.packageVersion(package_name)
        package_version = str(version_print).split("]")[1].strip().replace("'", "")
    return package_version


def install_limma_if_missing() -> None:
    """Installs limma if it is not installed already."""
    _install_missing_r_packages(["BiocManager"])
    _install_missing_bioconductor_packages(["limma"])


def _install_missing_r_packages(packages: list[str]) -> None:
    for package in packages:
        if not rpackages.isinstalled(package):
            print(f"Installing R package {package} ...")
            utils = importr("utils")
            utils.chooseCRANmirror(ind=1)
            utils.install_packages(package)


def _install_missing_bioconductor_packages(packages: list[str]) -> None:
    for package in packages:
        if not rpackages.isinstalled(package):
            print(f"Installing R package with BiocManager {package} ...")
            biocm = importr("BiocManager")
            biocm.install(package)
