from setuptools import setup, find_packages


setup(
    name='CORONAnet',
    version='0.0.1',
    description="Repository for deep learning scripts for predicting proton intensity "
    + "of CME events using coronagraph frames as input",
    author="Peter Thomas",
    author_email="pthomas2019@my.fit.edu",
    packages=find_packages(),
    install_required=[
        "numpy",
        "pandas"
        "pyrallis",
        "tqdm",
        "matplotlib",
        "tensorflow>=2.4.0",
    ]
)
