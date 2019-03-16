import setuptools

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hypergrad",
    version="0.1",
    author="Atılım Güneş Baydin",
    author_email="",
    description="Hypergradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gbaydin/hypergradient-descent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
