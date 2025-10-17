from setuptools import setup, find_packages

setup(
    name='supervision',
    version='0.27.0rc1',
    packages=find_packages(include=['supervision*'], exclude=['test*', 'docs*', 'examples*']),
    install_requires=[
        "numpy>=1.21.2",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "pyyaml>=5.3",
        "defusedxml>=0.7.1",
        "pillow>=9.4",
        "requests>=2.26.0",
        "tqdm>=4.62.3",
        "opencv-python>=4.5.5.64"
    ],
)
