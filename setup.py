from setuptools import setup, find_packages

setup(
    name="bendcontourpy",
    version="0.1.0",
    author="Henry Bell",
    author_email="hebell@stanford.edu",
    description="Package for TEM bend contour analysis",
    long_description = open("README.md").read(),
    long_description_content_type="text/markdown",
    url = "https://github.com/hebell/bendcontourpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.5.0", 
        "cupy>=10.0.0",
    ],
)