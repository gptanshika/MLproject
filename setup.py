from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [ i.replace('\n',"") for i in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
        return requirements


setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Anshika',
    author_email = 'anshikagupta505@gmail.com',
    packages=find_packages(),
    requires=get_requirements('requirements.txt')
)