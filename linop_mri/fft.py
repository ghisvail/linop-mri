"""Module implementing FFT related operators."""
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

try:
    from pyfftw.interfaces import numpy_fft as fft
except ImportError:
    from numpy import fft


__all__ = ('FFTOperator', 'FFTShiftOperator', 'IFFTOperator',
           'IFFTShiftOperator')


class FFTOperator(LinearOperator):
    
    """The forward Fast Fourier Transform operator."""
    
    def __init__(self, shapein, axes=None, s=None, **kwargs):
        
        tranpose_of = kwargs.pop('transpose_of',
            IFFTOperator(shapein=shapein,
                         axes=axes,
                         s=s,
                         transpose_of=self,
                         **kwargs))
        
        def matvec(x):
            return fft.fftn(x.reshape(shapein), axes=axes, s=s)
        
        super(FFTOperator, self).__init__(
            nargin=np.prod(shapein),
            nargout=np.prod(shapein),
            symmetric=False,
            matvec=matvec,
            dtype=np.complex,
            transpose_of=transpose_of,
            **kwargs)


class IFFTOperator(LinearOperator):
    
    """The adjoint Fast Fourier Transform operator."""
    
    def __init__(self, shapein, axes=None, s=None, **kwargs):
        
        tranpose_of = kwargs.pop('transpose_of',
            FFTOperator(shapein=shapein,
                        axes=axes,
                        s=s,
                        transpose_of=self,
                        **kwargs))
        
        def matvec(x):
            return fft.ifftn(x.reshape(shapein), axes=axes, s=s)
        
        super(FFTOperator, self).__init__(
            nargin=np.prod(shapein),
            nargout=np.prod(shapein),
            symmetric=False,
            matvec=matvec,
            dtype=np.complex,
            transpose_of=transpose_of,
            **kwargs)


class FFTShiftOperator(LinearOperator):
    
    """The forward half-space swap operator."""
    
    def __init__(self, shapein, axes=None, **kwargs):
        
        tranpose_of = kwargs.pop('transpose_of',
            IFFTShiftOperator(shapein=shapein,
                              axes=axes,
                              transpose_of=self,
                              **kwargs))
        
        def matvec(x):
            return fft.fftshift(x.reshape(shapein), axes=axes)
        
        super(FFTShiftOperator, self).__init__(
            nargin=np.prod(shapein),
            nargout=np.prod(shapein),
            symmetric=False,
            matvec=matvec,
            transpose_of=transpose_of,
            **kwargs)


class IFFTShiftOperator(LinearOperator):
    
    """The adjoint half-space swap operator."""
    
    def __init__(self, shapein, axes=None, **kwargs):
        
        tranpose_of = kwargs.pop('transpose_of',
            FFTShiftOperator(shapein=shapein,
                             axes=axes,
                             transpose_of=self,
                             **kwargs))
        
        def matvec(x):
            return fft.ifftshift(x.reshape(shapein), axes=axes)
                             
        super(IFFTShiftOperator, self).__init__(
            nargin=np.prod(shapein),
            nargout=np.prod(shapein),
            symmetric=False,
            matvec=matvec,
            transpose_of=transpose_of,
            **kwargs)
