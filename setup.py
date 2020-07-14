from setuptools import setup, find_packages

setup(name='csgan',
      version='0.1',
      description='Conditional GAN for speed simulation in street scenes',
      author='Vishal Sowrirajan',
      license='MIT',
      packages=['csgan', 'csgan.data', 'scripts'],
      zip_safe=False)
