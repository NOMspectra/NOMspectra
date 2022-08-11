from setuptools import setup
from setuptools import find_packages

import os
here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "nhsmass", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name='nhsmass',
    version=version["__version__"],
    license='GPLv3',
    description = 'Lib for working with high resolution mass spectra',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Volikov Alexander, Rukhovich Gleb',
    url = 'https://github.com/nhsmass/nhsmass',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'matplotlib>=3.1.2',
        'numpy>=1.18.1',
        'pandas>=0.25.3',
        'scipy>=1.4.1',
        'seaborn>=0.10.0',
        'tqdm>=4.43.0',
        'frozendict>=2.3.4',
        'networkx>=2.7.1'
	],
    extras_require={"dev": ["pytest",
                            "sphinx>=4.4.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
  ],
)