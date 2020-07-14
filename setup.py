from setuptools import setup, find_packages

with open('requirements.txt') as f:
    if f.startswith("#"):
        pass
    requirements = f.read().splitlines()

setup(name='csgan',
      version='0.1',
      description='Conditional GAN for speed simulation in street scenes',
      author='Vishal Sowrirajan',
      license='MIT',
      install_requires=requirements,
      packages=['csgan', 'csgan.data', 'scripts'],
      zip_safe=False)
