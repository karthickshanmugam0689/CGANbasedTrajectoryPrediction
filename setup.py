from setuptools import setup

setup(name='csgan',
      version='0.1',
      description='Conditional GAN for speed simulation in street scenes',
      author='Vishal Sowrirajan',
      packages=['csgan', 'csgan.data', 'scripts'],
      zip_safe=False)