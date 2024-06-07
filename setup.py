from setuptools import setup, find_packages
from gener import __version__, __authors__

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gener',
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    author=__authors__,
    keywords='natural-language-processing named-entity-recognition synthetic-data-generation machine-learning nlp-library large-langauge-models',
    description='Generative NER: Synthetic NER Data Creation Tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cutlerci/GeNER',
)