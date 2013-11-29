"""Setup script."""
#This file is part of linop-mri.
#
#linop-mri is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#linop-mri is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with linop-mri.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages


DESCRIPTION = 'linear operators for MRI reconstruction'
LONG_DESCRIPTION = open('README.rst').read()

setup(name='linop-mri',
      version='0.1-dev',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author='Ghislain Vaillant',
      author_email='ghisvail@gmail.com',
      url='https://github.com/ghisvail/linop-mri',
      packages=find_packages(),
      license='GPLv3',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
      ],
      install_requires = ['linop', 'numpy', 'setuptools'],
)
