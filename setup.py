import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = file.read()

setuptools.setup(
    name="Named Entity Recognition and Its Application to Phishing Detection",
    version="1.0",
    description="Additional source files to the bachelor thesis",
    author="Tomáš Pop",
    author_email="tom.pop16@seznam.cz",
    url="https://github.com/poptomas/ner-pd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[requirements],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
