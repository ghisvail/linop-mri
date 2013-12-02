""" module implementing NFFT related operators
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

from linop import LinearOperator


__all__ = ('NFFTOperator',)


class NFFTOperator(LinearOperator):
    
    """NFFTOperator."""
    
    def __init__(self, plan, **kwargs):
        
        nargin = np.prod(plan.N)
        nargout = plan.M
        matvec = lambda x: return plan.forward(f_hat=x)
        matvec_transp = lambda x: return plan.adjoint(f=x)        
        dtype = np.complex

        super(NFFTOperator, self).__init__(nargin,
                                           nargout,
                                           symmetric=False,
                                           matvec=matvec,
                                           matvec_transp=matvec_transp,
                                           dtype=np.complex128,
                                           **kwargs)
