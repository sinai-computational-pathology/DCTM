import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from DCTM import bernstein
from scipy import optimize
from lifelines.utils import concordance_index
gompertz_extra_inter = np.log(-np.log(0.5))
logistic_extra_inter = 0
extra_inter = {
    'logistic': logistic_extra_inter,
    'gompertz': gompertz_extra_inter
}

def c_index_t(risk, true_time, true_event, time_horizon):
      '''
      risk: log cumulative hazard at time_horizon, np array
      true_time: event times, np array
      true_event: 1-event, 0-censored, np array
      time_horizon: time at which to evaluate c-index, scalar
      '''
      dummy_event = true_event.copy()
      dummy_event[true_time>time_horizon] = 0
      return concordance_index(true_time, -risk, event_observed=dummy_event)

class get_family_function(object):
    def __init__(self, family):
        self.family = family
        self.familyfuns = {
            'logistic':
            {
                'logpdf': lambda x: torch.log(torch.sigmoid(x)) + toch.log(1-torch.sigmoid(x)),
                'pdf': lambda x: torch.sigmoid(x)*(1-torch.sigmoid(x)),
                'cdf': torch.sigmoid,
                'logsurv': lambda x: torch.log(1-torch.sigmoid(x))
            },
            'gompertz':
            {
                'logpdf': lambda x: x-torch.exp(x),
                'pdf': lambda x: torch.exp(x-torch.exp(x)),
                'cdf': lambda x: 1-torch.exp(-torch.exp(x)),
                'logsurv':lambda x: -torch.exp(x)
            }
        }
    def pdf(self, x):
        return self.familyfuns[self.family]['pdf'](x)
    def cdf(self, x):
        return self.familyfuns[self.family]['cdf'](x)
    def logpdf(self, x):
        return self.familyfuns[self.family]['logpdf'](x)
    def logsurv(self, x):
        return self.familyfuns[self.family]['logsurv'](x)

class bernstein_coefficients(nn.Module):
    def __init__(self):
        super(bernstein_coefficients, self).__init__()
    
    def forward(self, x):
        a = x[:,0]
        b = x[:,1:]
        b = F.softplus(b)
        c = torch.cat([a.view(-1,1), b], dim=1)
        output = torch.cumsum(c, dim=1)
        return output

class DCTM_general(nn.Module):
    def __init__(self, input_features, basis_features, family=''):
        super(DCTM_general, self).__init__()
        self.input_features = input_features
        self.basis_features = basis_features
        self.linear = nn.Linear(input_features, basis_features)
        self.bernstein_basis = bernstein.bernstein_basis(basis_features-1)
        self.coeffs = bernstein_coefficients()
        self.family = get_family_function(family)
        self.extra_inter = extra_inter[family]
    
    def _make_time_basis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.run(time)

    def _make_time_antibasis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.der(time)
    
    def _find_tte_sample(self, coeffs):
        '''
        coeffs are 1D of size P+1
        returns single estimate based on root finder
        '''
        # Create function
        f = self.bernstein_basis._create_tte_fun(coeffs)
        # Find root
        if f(0.) * f(1.) > 0:
            # If no crossing zero
            if f(0.) > 0:
                return 0.
            else:
                return 1.
        else:
            sol = optimize.root_scalar(f, bracket=[0., 1.], method='bisect')
            # Make sure it's valid
            if sol.converged or sol.flag == '':
                return sol.root
            else:
                return np.nan
    
    def _find_tte_batch(self, coeffs):
        ttes = np.empty(coeffs.shape[0])
        for i in range(coeffs.shape[0]):
            ttes[i] = self._find_tte_sample(coeffs[i])
        return ttes
    
    def forward(self, input, time):
        # Transform to P+1
        input = self.linear(input)
        # Produce increasing coefficients
        coeffs = self.coeffs(input)
        # Produce the time basis
        basis = self._make_time_basis(time)
        basis = basis.to(coeffs.device)
        # Dot products
        output = torch.sum(coeffs * basis, dim=1)
        return output
    
    def tte(self, input):
        '''
        tte as a 1D array
        '''
        input = self.linear(input)
        coeffs = self.coeffs(input)
        coeffs = coeffs.detach().cpu().numpy()
        # Root finder iteratively
        ttes = self._find_tte_batch(coeffs)
        return ttes
    
    def nllloss(self, input, time, event):
        input = self.linear(input)
        coeffs = self.coeffs(input)
        basis = self._make_time_basis(time)
        basis = basis.to(coeffs.device)
        antibasis = self._make_time_antibasis(time)
        antibasis = antibasis.to(coeffs.device)
        z = torch.sum(coeffs * basis, dim=1)
        a1 = self.family.cdf(z)# <-- F(h(y))
        a2 = self.family.pdf(z)# <-- f(h(y))
        a3 = torch.sum(coeffs * antibasis, dim=1) # <-- h'(y)
        nll = ( event*(torch.log(torch.clamp(a2, min=1e-10)) + torch.log(a3)) + (1-event)*(torch.log(torch.clamp(1-a1, min=1e-10))) ) *(-1)
        return nll.mean()

class DCTM_general_shift(nn.Module):
    def __init__(self, input_features, basis_features, family=''):
        super(DCTM_general_shift, self).__init__()
        self.input_features = input_features
        self.basis_features = basis_features
        self.w_input = nn.Linear(input_features, 1, bias=False)
        self.w_basis = nn.Parameter(torch.Tensor(1, basis_features))
        self.bernstein_basis = bernstein.bernstein_basis(basis_features-1)
        self.coeffs = bernstein_coefficients()
        self.family = get_family_function(family)
        # Initialize
        self.w_basis.data.uniform_(-1, 1)
        self.extra_inter = extra_inter[family]
    
    def _make_time_basis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.run(time)

    def _make_time_antibasis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.der(time)
    
    def _find_tte_sample(self, coeffs, inter):
        '''
        coeffs are 1D of size P+1
        returns single estimate based on root finder
        '''
        # Create function
        f = self.bernstein_basis._create_tte_fun(coeffs, intercept=inter+self.extra_inter)
        # Find root
        if f(0.) * f(1.) > 0:
            # pdb.set_trace()
            # If no crossing zero
            if f(0.) > 0:
                return 0.
            else:
                return 1.
        else:
            sol = optimize.root_scalar(f, bracket=[0., 1.], method='bisect')
            # Make sure it's valid
            if sol.converged or sol.flag == '':
                return sol.root
            else:
                return np.nan
    
    def _find_tte_batch(self, coeffs, inter):
        ttes = np.empty(coeffs.shape[0])
        # pdb.set_trace()
        for i in range(coeffs.shape[0]):
            ttes[i] = self._find_tte_sample(coeffs[i], inter[i].item())
        return ttes
    
    def forward(self, input, time):
        # X (B,D) --> 1 (B,1)
        shift = self.w_input(input).squeeze(1)
        # Produce increasing coefficients
        coeffs = self.coeffs(self.w_basis)
        # Produce the time basis
        basis = self._make_time_basis(time)
        basis = basis.to(input.device)#(B,P+1)
        # Dot products
        output = torch.sum(coeffs * basis, dim=1) + shift#B,1
        return output
    
    def tte(self, input):
        '''
        tte as a 1D array
        '''
        shift = self.w_input(input).squeeze(1)
        coeffs = self.coeffs(self.w_basis)
        coeffs = coeffs.detach().cpu().repeat(shift.size(0),1).numpy()
        # Root finder iteratively
        ttes = self._find_tte_batch(coeffs, -shift)
        return ttes
    
    def nllloss(self, input, time, event):
        shift = self.w_input(input).squeeze(1)
        coeffs = self.coeffs(self.w_basis)
        basis = self._make_time_basis(time)
        basis = basis.to(input.device)
        antibasis = self._make_time_antibasis(time)
        antibasis = antibasis.to(input.device)
        z = torch.sum(coeffs * basis, dim=1) + shift
        a1 = self.family.cdf(z)# <-- F(h(y))
        a2 = self.family.pdf(z)# <-- f(h(y))
        a3 = torch.sum(coeffs * antibasis, dim=1) # <-- h'(y)
        nll = ( event*(torch.log(torch.clamp(a2, min=1e-10)) + torch.log(a3)) + (1-event)*(torch.log(torch.clamp(1-a1, min=1e-10))) ) *(-1)
        return nll.mean()
        
class DCTM_general_shift_scale(nn.Module):
    def __init__(self, input_features, basis_features, family=''):
        super(DCTM_general_shift_scale, self).__init__()
        self.input_features = input_features
        self.basis_features = basis_features
        self.wshift_input = nn.Linear(input_features, 1)
        self.wscale_input = nn.Linear(input_features, 1)
        self.w_basis = nn.Parameter(torch.Tensor(1, basis_features))
        self.bernstein_basis = bernstein.bernstein_basis(basis_features-1)
        self.coeffs = bernstein_coefficients()
        self.family = get_family_function(family)
        # Initialize
        self.w_basis.data.uniform_(-1, 1)
        self.extra_inter = extra_inter[family]
    
    def _make_time_basis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.run(time)

    def _make_time_antibasis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.der(time)
    
    def _find_tte_sample(self, coeffs, inter):
        '''
        coeffs are 1D of size P+1
        returns single estimate based on root finder
        '''
        # Create function
        f = self.bernstein_basis._create_tte_fun(coeffs, intercept=(inter+self.extra_inter))
        # Find root
        if f(0.) * f(1.) > 0:
            # If no crossing zero
            if f(0.) > 0:
                return 0.
            else:
                return 1.
        else:
            sol = optimize.root_scalar(f, bracket=[0., 1.], method='bisect')
            # Make sure it's valid
            if sol.converged or sol.flag == '':
                return sol.root
            else:
                return np.nan
    
    def _find_tte_batch(self, coeffs, inter):
        ttes = np.empty(coeffs.shape[0])
        for i in range(coeffs.shape[0]):
            ttes[i] = self._find_tte_sample(coeffs[i], inter[i].item())
        return ttes
    
    def forward(self, input, time):
        # X (B,D) --> 1 (B,1)
        shift = self.wshift_input(input).squeeze(1)
        scale = F.softplus(self.wscale_input(input)).squeeze(1)
        # Produce increasing coefficients
        coeffs = self.coeffs(self.w_basis)
        # Produce the time basis
        basis = self._make_time_basis(time)
        basis = basis.to(input.device)#(B,P+1)
        # Dot products
        output = torch.sum(coeffs * basis, dim=1) * scale + shift#B,1
        return output
    
    def tte(self, input):
        '''
        tte as a 1D array
        '''
        shift = self.wshift_input(input).squeeze(1)
        scale = F.softplus(self.wscale_input(input))
        scale = scale.detach().cpu().numpy()
        coeffs = self.coeffs(self.w_basis)
        coeffs = coeffs.detach().cpu().repeat(shift.size(0),1).numpy()
        # Root finder iteratively
        ttes = self._find_tte_batch(coeffs * scale, -shift)
        return ttes
    
    def nllloss(self, input, time, event):
        shift = self.wshift_input(input).squeeze(1)
        scale = F.softplus(self.wscale_input(input)).squeeze(1)
        coeffs = self.coeffs(self.w_basis)
        basis = self._make_time_basis(time)
        basis = basis.to(input.device)
        antibasis = self._make_time_antibasis(time)
        antibasis = antibasis.to(input.device)
        z = torch.sum(coeffs * basis, dim=1) * scale + shift
        a1 = self.family.cdf(z)# <-- F(h(y))
        a2 = self.family.pdf(z)# <-- f(h(y))
        a3 = torch.sum(coeffs * antibasis, dim=1)*scale # <-- h'(y)*scale
        nll = ( event*(torch.log(torch.clamp(a2, min=1e-10)) + torch.log(a3)) + (1-event)*(torch.log(torch.clamp(1-a1, min=1e-10))) ) *(-1)
        return nll.mean()

class DCTM_baseline(nn.Module):
    def __init__(self, basis_features, family=''):
        super(DCTM_baseline, self).__init__()
        self.basis_features = basis_features
        self.w_basis = nn.Parameter(torch.Tensor(1, basis_features))
        self.bernstein_basis = bernstein.bernstein_basis(basis_features-1)
        self.coeffs = bernstein_coefficients()
        self.family = get_family_function(family)
        # Initialize
        self.w_basis.data.uniform_(-1, 1)
        self.extra_inter = extra_inter[family]
    
    def _make_time_basis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.run(time)
    
    def _make_time_antibasis(self, time):
        '''
        Time is a B or Bx1 ndarray
        Output is Bx(P+1)
        '''
        return self.bernstein_basis.der(time)
    
    def _find_tte_sample(self, coeffs, inter=0):
        '''
        coeffs are 1D of size P+1
        returns single estimate based on root finder
        '''
        # Create function
        f = self.bernstein_basis._create_tte_fun(coeffs, intercept=inter+self.extra_inter)
        # Find root
        if f(0.) * f(1.) > 0:
            # pdb.set_trace()
            # If no crossing zero
            if f(0.) > 0:
                return 0.
            else:
                return 1.
        else:
            sol = optimize.root_scalar(f, bracket=[0., 1.], method='bisect')
            # Make sure it's valid
            if sol.converged or sol.flag == '':
                return sol.root
            else:
                return np.nan
    
    def _find_tte_batch(self, coeffs):
        ttes = np.empty(coeffs.shape[0])
        for i in range(coeffs.shape[0]):
            ttes[i] = self._find_tte_sample(coeffs[i])
        return ttes
    
    def forward(self, time):
        # Produce increasing coefficients
        coeffs = self.coeffs(self.w_basis)
        # Produce the time basis
        basis = self._make_time_basis(time)
        basis = basis.to(self.w_basis.device)#(B,P+1)
        # Dot products
        output = torch.sum(coeffs * basis, dim=1)#B,1
        return output
    
    def tte(self,batch_size):
        '''
        tte as a 1D array
        '''
        coeffs = self.coeffs(self.w_basis)
        coeffs = coeffs.detach().cpu().repeat(batch_size,1).numpy()
        # Root finder iteratively
        ttes = self._find_tte_batch(coeffs)
        return ttes
    
    def nllloss(self, time, event):
        coeffs = self.coeffs(self.w_basis)
        basis = self._make_time_basis(time)
        basis = basis.to(self.w_basis.device)
        antibasis = self._make_time_antibasis(time)
        antibasis = antibasis.to(self.w_basis.device)
        z = torch.sum(coeffs * basis, dim=1)
        a1 = self.family.cdf(z)# <-- F(h(y))
        a2 = self.family.pdf(z)# <-- f(h(y))
        a3 = torch.sum(coeffs * antibasis, dim=1) # <-- h'(y)
        nll = ( event*(torch.log(torch.clamp(a2, min=1e-10)) + torch.log(a3)) + (1-event)*(torch.log(torch.clamp(1-a1, min=1e-10))) ) *(-1)
        return nll.mean()

def get_model(dctm, F=None, B=None, family=None):
    if dctm == 'baseline':
        model = DCTM_baseline(B, family=family)
    elif dctm == 'general_shift':
        model = DCTM_general_shift(F, B, family=family)
    elif dctm == 'general_shift_scale':
        model = DCTM_general_shift_scale(F, B, family=family)
    elif dctm == 'general':
        model = DCTM_general(F, B, family=family)
    else:
        raise Exception(f'Model {dctm} not implemented')
    return model
