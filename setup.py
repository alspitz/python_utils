from setuptools import find_packages, setup

setup(
    name="python_utils",
    version="0.1.0",
    packages=find_packages(),
    author="Alex Spitzer",
    maintainer="Alex Spitzer",
    url="https://github.com/alspitz/python_utils",
    license=open("LICENSE", mode="r").read(),
    install_requires=(
      'numpy',
      'scipy',
      'matplotlib',
      'joblib',
    ),
)
