from setuptools import setup, find_packages


VERSION = "0.0.1"

packages = find_packages()
packages.append("msreport.rinterface.rscripts")

setup(
    name="msreport",
    version=VERSION,
    license="Apache v2",
    author="David M. Hollenstein",
    author_email="hollenstein.david@gmail.com",
    install_requires=[
        "maspy",
        "matplotlib",
        "numpy",
        "pandas",
        "pyteomics",
        "rpy2",
        "scipy",
        "seaborn",
        "sklearn",
        "statsmodels",
    ],
    extras_require={
        "test": ["pytest"],
    },
    python_requires=">=3.9",
    packages=packages,
    package_data={"msreport.rinterface.rscripts": ["*.r", "*.R"]},
    keywords=["proteomics", "mass spectrometry", "data analysis", "data processing"],
)
