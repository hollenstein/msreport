from setuptools import setup, find_packages


VERSION = "0.0.1"
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
        "scipy",
        "seaborn",
        "statsmodels",
        "xlsxwriter"  # Excel report
        "yaml"  # Excel report
        "rpy2"  # R interface
        "pytest",  # tests
    ],
    extras_require={
        "test": ["pytest"],
    },
    python_requires=">=3.9",
    packages=find_packages(
        where="msreport",
    ),
    keywords=["proteomics", "mass spectrometry", "data analysis", "data processing"],
)
