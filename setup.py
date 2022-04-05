from setuptools import setup, find_packages

setup(
   name='flexsim',
   packages=find_packages(),
   version='0.1',
   description='Set of tools for generation of realistic X-ray projections',
   author='V. Andriiashen',
   author_email='vladyslav.andriiashen@cwi.nl',
   url = 'https://github.com/vandriiashen/flexsim',
   package_dir={'flexsim': 'flexsim'}
)
