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
from linop import LinearOperator, IdentityOperator

try:
    from pyfftw.interfaces import numpy_fft as fft
except ImportError:
    from numpy import fft


__all__ = ('FFTOperator', 'FFTShiftOperator')


class FFTOperator(LinearOperator):

    """The Fourier transform operator."""

    def __init__(self, shapein, axes=None, s=None, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        transpose_of = kwargs.get('transpose_of', None)

        nargin = nargout = np.prod(shapein)
        dtype = np.complex

        def matvec(x):
            return fft.fftn(x.reshape(shapein), axes=axes, s=s)

        if transpose_of is None:
            H = IFFTOperator(shapein=shapein, axes=axes, s=s, dtype=dtype,
                             transpose_of=self, **kwargs)

        super(FFTOperator, self).__init__(nargin, nargout, symmetric=False,
            matvec=matvec, dtype=dtype, transpose_of=H, **kwargs)

    # TODO: think about decorating unitary operators
    def _LinearOperator__mul_linop(self, op):
        if op.H is self:
            return IdentityOperator(self.nargin, dtype=self.dtype)
        else:
            super(FFTOperator, self)._LinearOperator__mul_linop(op)


class IFFTOperator(LinearOperator):

    """The Fourier transform operator."""

    def __init__(self, shapein, axes=None, s=None, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        transpose_of = kwargs.get('transpose_of', None)

        nargin = nargout = np.prod(shapein)
        dtype = np.complex

        def matvec(x):
            return fft.ifftn(x.reshape(shapein), axes=axes, s=s)

        if transpose_of is None:
            H = FFTOperator(shapein=shapein, axes=axes, s=s, dtype=dtype,
                            transpose_of=self, **kwargs)

        super(IFFTOperator, self).__init__(nargin, nargout, symmetric=False,
            matvec=matvec, dtype=dtype, **kwargs)

    # TODO: think about decorating unitary operators
    def _LinearOperator__mul_linop(self, op):
        if op.H is self:
            return IdentityOperator(self.nargin, dtype=self.dtype)
        else:
            super(IFFTOperator, self)._LinearOperator__mul_linop(op)


class FFTShiftOperator(LinearOperator):
    def __init__(self, shapein, axes=None, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')

        nargin = nargout = np.prod(shapein)
        dtype = kwargs.pop('dtype', np.complex)

        def matvec(x):
            return fft.fftshift(x.reshape(ndshape), axes=axes)

        def matvec_transp(x):
            return fft.ifftshift(x.reshape(ndshape), axes=axes)

        super(FFTShiftOperator, self).__init__(
            nargin, nargout, symmetric=False, matvec=matvec,
            matvec_transp=matvec_transp, dtype=dtype, **kwargs)
