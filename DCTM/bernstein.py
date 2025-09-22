import numpy as np
import torch
from scipy.special import comb
import pdb

def create_base_fun(n, v):
    b = comb(n, v, exact=True)
    return lambda x: b * x**v * (1-x)**(n-v)

def create_base_fun_string(n, v):
    b = comb(n, v, exact=True)
    funstr = '{} * x**{} * (1-x)**{}'.format(b, v, n-v)
    # print(funstr)
    return funstr

def create_base_der(n, v):
    b1 = comb(n-1, v-1, exact=True)
    b2 = comb(n-1, v, exact=True)
    return lambda x: n * ((b1 * x **(v-1) * (1-x)**(n-v)) - (b2 * x**v * (1-x)**(n-v-1)))

class bernstein_basis(object):
    '''
    Generates the n bernstein basis polynomials
    '''
    def __init__(self, n):
        self.n = n
        self.funcs = []
        self.strings = []
        self.derivs = []
        for v in range(n+1):
            self.funcs.append(create_base_fun(n, v))
            self.strings.append(create_base_fun_string(n, v))
            self.derivs.append(create_base_der(n, v))
    
    def _run_float(self, t):
        out = np.empty(self.n+1)
        for v in range(self.n+1):
            out[v] = self.funcs[v](t)
        return torch.tensor(out).view(1, -1)
    def _run_ndarray(self, t):
        out = np.empty((t.shape[0], self.n+1))
        for v in range(self.n+1):
            out[:,v] = self.funcs[v](t)
        return torch.tensor(out)
    def run(self, t):
        if isinstance(t, int) or isinstance(t, float):
            return self._run_float(t)
        elif isinstance(t, np.ndarray):
            if len(t.shape) > 2 or (len(t.shape) == 2 and t.shape[1] != 1):
                raise Exception('Ndarray should be 1D or have second dimension of size 1')
            return self._run_ndarray(t)
        else:
            raise Exception('Time should either be an int, a float or a np.ndarray')
    
    def _der_float(self, t):
        out = np.empty(self.n+1)
        for v in range(self.n+1):
            out[v] = self.derivs[v](t)
        return torch.tensor(out).view(1, -1)
    def _der_ndarray(self, t):
        out = np.empty((t.shape[0], self.n+1))
        for v in range(self.n+1):
            ans = self.derivs[v](t)
            if np.isnan(ans).sum() > 0:
                raise Exception('NANs when calculating derivative of bernstein basis')
            out[:,v] = ans
        return torch.tensor(out)
    def der(self, t):
        if isinstance(t, int) or isinstance(t, float):
            return self._der_float(t)
        elif isinstance(t, np.ndarray):
            # Clip to 1e-10
            t = np.clip(t, 1e-7, 1-1e-7)
            if len(t.shape) > 2 or (len(t.shape) == 2 and t.shape[1] != 1):
                raise Exception('Ndarray should be 1D or have second dimension of size 1')
            return self._der_ndarray(t)
        else:
            raise Exception('Time should either be an int, a float or a np.ndarray')
    
    def _create_tte_fun(self, coeffs, intercept=0):
        full = []
        for v in range(self.n+1):
            full.append('({} * {})'.format(self.strings[v], coeffs[v]))
        full = ' + '.join(full)
        if intercept != 0.:
            full += ' - {}'.format(intercept)
        full = 'def f(x): return {}'.format(full)
        d = {}
        exec(full, d)
        return d['f']
