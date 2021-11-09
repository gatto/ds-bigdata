import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    # $ pip install sampleproject https://pypi.org/project/sampleproject/
    name="extractbda",
    version="0.3",
    description="Extract csv data for 2021 BDA Project Bike Sharing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gatto/ds-bigdata",
    author="Fabio Michele Russo",
    classifiers=[
        # 3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        # Specify the Python versions you support here. In particular, ensure that you indicate you support Python 3. These classifiers are *not* checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="datascience, extract, etl",
    package_dir={"extractbda": "src"},
    packages=["extractbda"],
    python_requires=">=3.7, <4",
    install_requires=["pandas", "attrs"],
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={
        "extractbda": ["data/*.csv"],
    },
    project_urls={
        "Bug Reports": "https://github.com/gatto/ds-bigdata/issues",
        "Source": "https://github.com/gatto/ds-bigdata/",
    },
)
