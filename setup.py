import os
import sys

from setuptools import setup, find_packages

def readme():
	with open('README.rst') as f:
		return f.read()

exec(open(os.path.join('saleae', 'version.py')).read())

requires = ["enum34"]

setup(
		name='saleae',
		version=__version__,
		#
		packages=find_packages(exclude=['tests']),
		#
		description="Library to control a Saleae",
		long_description=readme(),
		#
		url="https://github.com/ppannuto/python-saleae",
		#
		author="Pat Pannuto",
		author_email="pat.pannuto+saleae@gmail.com",
		#
		license="MIT",
		#
		classifiers=[
			"Development Status :: 3 - Alpha",
			"Intended Audience :: Developers",
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: MIT License",
			"Operating System :: OS Independent",
			"Topic :: Scientific/Engineering",
			"Topic :: Software Development :: Embedded Systems",
			"Topic :: Utilities",
			],
		#
		keywords='string formatting',
		#
		install_requires=requires,
		include_package_data=True,
		#
		#test_suite='saleae.tests',
		#
		#entry_points = {
		#    'console_scripts':['saleae = saleae:console']
		#},
		)

