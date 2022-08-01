from setuptools import setup
from setuptools import find_packages

import os
here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "nhsmasslib", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name='nhsmasslib',
    version=version["__version__"],
    license='GPLv3',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Volikov Alexander, Rukhovich Gleb',
    url = 'https://github.com/nhsmasslib/nhsmasslib',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
        'matplotlib>=3.1.2',
        'numpy>=1.18.1',
        'pandas>=0.25.3',
        'scipy>=1.4.1',
        'seaborn>=0.10.0',
        'tqdm>=4.43.0',
        'frozendict>=2.3.4'
	],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
  ],
)