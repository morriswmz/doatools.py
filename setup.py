from setuptools import setup, find_packages

setup(
    name='doatools',
    version='0.1.0',
    description='A collection of tools for DOA estimation related research.',
    # long_description='',
    # url='',
    author='Mianzhi Wang',
    # author_email='',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.14.0',
        'scipy>=1.1.0',
        'matplotlib>=2.1.0'
    ],
    zip_safe=False
)
