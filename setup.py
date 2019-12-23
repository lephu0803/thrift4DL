from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thrift4DL",
    version="2.0.0", # New version
    author="congvm",
    include_package_data=True,
    description="Thrift for Deep Learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=["pytest", "mock"],
    test_suite="pytest",
)