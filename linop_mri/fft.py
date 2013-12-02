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


__all__ = ('FFTOperator', 'FFTShiftOperator')






class FFTOperator(LinearOperator):
    def __init__(self, nargin, nargout, ndshape, axes=None, s=None,
                 **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')
        if 'dtype' in kwargs:
            kwargs.pop('dtype')

        def matvec(x):
            return fft.fftn(x.reshape(ndshape), axes=axes, s=s)

        def matvec_transp(x):
            return fft.ifftn(x.reshape(ndshape), axes=axes, s=s)

        super(FFTOperator, self).__init__(
            nargin, nargout, symmetric=False, matvec=matvec,
            matvec_transp=matvec_transp, dtype=dtype, **kwargs)


class FFTShiftOperator(LinearOperator):
    def __init__(self, nargin, nargout, ndshape, axes=None, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')
        dtype = kwargs.pop('dtype', np.complex)

        def matvec(x):
            return fft.fftshift(x.reshape(ndshape), axes=axes)

        def matvec_transp(x):
            return fft.ifftshift(x.reshape(ndshape), axes=axes)

        super(FFTShiftOperator, self).__init__(
            nargin, nargout, symmetric=False, matvec=matvec,
            matvec_transp=matvec_transp, dtype=dtype, **kwargs)
