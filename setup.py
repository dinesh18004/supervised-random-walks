from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='supervised-random-walks',
    version='0.0.0',

    description='Supervised Random Walks Algorithm',

    url='https://github.com/bmulvihill/supervised-random-walks',
    author='Bryan Mulvihill',
    author_email='mulvihill.bryan@gmail.com',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Research',
        'Topic :: Computer Science :: Algorithms',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='supervised random walks, data mining, algorithm, link prediction',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'scipy'],

    extras_require={
        'dev': ['check-manifest', 'nose'],
        'test': ['coverage'],
    },
)
