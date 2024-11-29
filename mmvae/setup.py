from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='WF-VAE',
    version='1.0.0',
    description='WF-VAE',
    author='qqingzheng',
    author_email='2533221180@qq.com',
    packages=find_packages(),
    install_requires=required
)