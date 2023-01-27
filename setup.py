from setuptools import setup, find_packages


VERSION = "0.0.6"

packages = find_packages()
packages.append("msreport.rinterface.rscripts")
packages.append("msreport_scripts.benchmark")
packages.append("msreport_scripts.excel_report")

setup(
    name="msreport",
    version=VERSION,
    license="Apache v2",
    author="David M. Hollenstein",
    author_email="hollenstein.david@gmail.com",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pyteomics",
        "rpy2>=3.5.3",
        "scipy",
        "seaborn",
        "sklearn",
        "statsmodels",
        "xlsxreport>=0.0.5",
    ],
    extras_require={
        "test": ["pytest"],
    },
    python_requires=">=3.9",
    packages=packages,
    package_data={"msreport.rinterface.rscripts": ["*.r", "*.R"]},
    keywords=["proteomics", "mass spectrometry", "data analysis", "data processing"],
)
