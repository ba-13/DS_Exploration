from setuptools import find_packages, setup
from typing import List


def get_requirements(requirements_file: str) -> List:
    requirements = []
    with open(requirements_file) as f:
        requirements = f.readlines()
    requirements = [x.strip() for x in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements


setup(
    name="ml_project",
    version="0.0.1",
    author="ba-13",
    author_email="banshuman20@iitk.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("./requirements.txt"),
)
