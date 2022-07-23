from distutils.core import setup

setup(
    name='nhsmasslib',
    version='0.1.3',
    packages=['nhsmasslib',],
    license='GPLv3',
    description='A small lib for treatment high-resolution mass spectrum',
    long_description=open('README.md').read(),
    author='Volikov Alexander, Rukhovich Gleb',
    url = 'https://github.com/nhsmasslib/nhsmasslib',
    download_url = 'https://github.com/nhsmasslib/nhsmasslib/archive/refs/tags/v0.1.3.tar.gz',
    install_requires=[
        'matplotlib>=3.1.2',
        'numpy>=1.18.1',
        'pandas>=0.25.3',
        'scipy>=1.4.1',
        'seaborn>=0.10.0',
        'tqdm>=4.43.0',
	],
)