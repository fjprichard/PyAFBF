import os
import re
from setuptools import setup, find_packages


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open('afbf/__init__.py').read()).group(1)

setup(
    # name of the package
    name='PyAFBF',
    # You can specify all the packages manually or use the find_package
    # function
    packages=find_packages(),
    # See PEP440 for defining a proper version number
    version=__version__,
    # Small description of the package
    description='Brownian texture simulation',
    # Long description
    long_description=(read('README.rst') + '\n\n' +
                      read('AUTHORS.rst') + '\n\n'),
    # Project home page:
    url='',
    # license, author and author email
    license='GNU GPL, Version 3',
    author='Frederic Richard',
    author_email='frederic.richard@univ-amu.fr',
    # If any packages contains data which are not python files, include them
    # package_data={'myapp': 'data'},
    install_requires=['numpy>=1.19.2', 'matplotlib>=3.3.2', 'scipy>=1.5.2'],
    # classifiers is needed for uploading package on pypi.
    # The list of classifiers elements can be found at :
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Users',
        'Natural Language :: English',
        'License :: GNU Public Licence, Version 3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: image texture, random field, simulation',
    ],
    # What does your project relate to?
    keywords={'image texture', 'Anisotropic fractional Brownian field',
              'simulation'},
    # Platforms on which the package can be installed:
    platforms='Linux, MacOSX, Windows',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'afbf=afbf:main',
        ],
    },
)
