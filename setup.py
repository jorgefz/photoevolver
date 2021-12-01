import setuptools
from photoevolver.version import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="photoevolver",
    version=__version__,
    author="Jorge Fernandez",
    author_email="Jorge.Fernandez-Fernandez@warwick.ac.uk",
    
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),

    description="Planetary evolver written in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jorgefz/photoevolver",
    license="LICENSE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
       "astropy",
       "numpy",
       "scipy",
       "matplotlib"
#       "Mors"
   ],
)
