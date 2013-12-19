"""Module implementing NFFT related operators."""

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

import numpy as np
from linop import LinearOperator


__all__ = ('AdjointNFFTOperator', 'ForwardNFFTOperator', 'NFFTOperator')


class ForwardNFFTOperator(LinearOperator):
    
    """The Non-uniform Fourier transform operator (forward)"""
    
    def __init__(self, plan, **kwargs):
        
        tranpose_of = kwargs.get('transpose_of',
            AdjointNFFTOperator(plan, transpose_of=self, **kwargs))
        
        super(ForwardNFFTOperator, self).__init__(
            nargin=np.prod(plan.N),
            nargout=plan.M,
            symmetric=False,
            matvec=lambda x: return plan.forward(f_hat=x),
            dtype=plan.dtype,
            transpose_of=transpose_of,
            **kwargs)
        

class AdjointNFFTOperator(LinearOperator)

    """The Non-uniform Fourier transform operator (adjoint)"""
    
     def __init__(self, plan, **kwargs):
         
        tranpose_of = kwargs.get('transpose_of',
            ForwardNFFTOperator(plan, transpose_of=self, **kwargs))
         
        super(AdjointNFFTOperator, self).__init__(
            nargin=np.prod(plan.N),
            nargout=plan.M,
            symmetric=False,
            matvec=lambda x: return plan.adjoint(f=x),
            dtype=plan.dtype,
            transpose_of=transpose_of,
            **kwargs)


NFFTOperator = ForwardNFFTOperator
