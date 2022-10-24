from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages


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
