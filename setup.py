from io import open

from setuptools import find_packages, setup

with open('torchrender/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = [
    'torch==1.0.1',
    'numpy==1.16.1',
    'opencv_python==3.4.5.20',
    'setuptools==40.8.0',
    'CVPubSubs==0.6.4',
]

setup(
    name='torchrender',
    version=version,
    description='',
    long_description=readme,
    author='SimLeek',
    author_email='josh.miklos@gmail.com',
    maintainer='SimLeek',
    maintainer_email='josh.miklos@gmail.com',
    url='https://github.com/SimLeek/torchrender',
    license='MIT',

    keywords=[
        'pytorch', 'shaders'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],

    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
