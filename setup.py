from distutils.core import setup

setup(
    name='nhsmasslib',
    version='0.1.1',
    packages=['nhsmasslib',],
    license='GPLv3',
    description='A small lib for treatment high-resolution mass spectrum',
    long_description=open('README.md').read(),
    author='Volikov Alexander, Rukhovich Gleb',
    install_requires=[
        'matplotlib>=3.1.2',
        'numpy>=1.18.1',
        'pandas>=0.25.3',
        'scikit-learn>=0.22.1',
        'scipy>=1.4.1',
        'seaborn>=0.10.0',
        'tqdm>=4.43.0',
	'networkx>=2.5',
	'pyvis>=0.2.1'
	],
)