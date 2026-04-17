from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

with open("version.txt", "r", encoding="utf-8") as fh:
    version = fh.read().strip()

setup(
    name="qasap",
    version=version,
    author="Erik Solhaug",
    description="Quick Analysis of Spectra and Profiles - Interactive spectral analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eriksolhaug/qasap",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qasap=qasap:main",
        ],
    },
    include_package_data=True,
)
