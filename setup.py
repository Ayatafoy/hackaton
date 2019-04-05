from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='src',
    version='0.1',
    description='Ottepel hackaton',
    classifiers=['Programming Language :: Python :: 3.6.6'],
    url='https://github.com/Ayatafoy/hackaton.git',
    author='Aleksey Romanov',
    author_email='aromanov@griddynamics.com',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False
)
