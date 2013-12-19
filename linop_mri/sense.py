""" module implementing SENSE related operators
"""
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

import pykrylov.cg as cg


class IterativeSENSE(cg.CG):
    def __init__(self, E, D=None, I=None, **kwargs):
        super(IterativeSENSE).__init__(op=None, **kwargs)

    def solve(m, **kwargs):
        super(IterativeSENSE).__init__(rhs=None, **kwargs)


def get_D(weights, ncoils):
    pass

def get_E():
    pass
    
def get_I():
    pass
