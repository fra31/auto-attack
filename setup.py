import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoattack",
    version="0.1",
    author="Francesco Croce, Matthias Hein",
    author_email="francesco.croce@uni-tuebingen.de",
    description="This package provides the implementation of AutoAttack.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fra31/auto-attack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)


