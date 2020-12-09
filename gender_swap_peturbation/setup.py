import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pronoun-transformation", # Replace with your own username
    version="0.0.1",
    author="Ben Siderowf, Shweta Chopra",
    description="Package for performing gender swaps on text data and measuring the corresponding \
    changes in performance of language based models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/choprashweta/classifier-bias/tree/main/gender_swap_peturbation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='==3.5',
)