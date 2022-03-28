from setuptools import setup, find_packages

setup_info = dict(
    name='masslib',
    version='0.1.0',
    author='Natural Humic Systems Laboratory',
    author_email='',
    url='https://github.com/nhslab/masslib',
    project_urls = {
        'Bug Tracker': 'https://github.com/nhslab/masslib/issues'
    },
    description='Small library for working with FTICR MS spectra',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Topic :: Scientific/Engineering :: Visualization',
    ],

    # Packages
    packages=['masslib'] + ['masslib.' + pkg for pkg
                            in find_packages('masslib')],
    
    # CSV data
    include_package_data=True,

    # Dependencies
    install_requires=open('requirements.txt').read().split('\n'),
    python_requires='>=3.6',
)

setup(**setup_info)