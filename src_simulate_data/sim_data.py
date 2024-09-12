from torch.utils.data import Dataset
from torch import tensor, rand, cdist, randn, exp as texp
from numpy.random import normal, uniform
from numpy import linspace, arange, meshgrid, array, exp, sin, concatenate, cos
from numpy import pi as npi, power
from numpy.linalg import det
from scipy.optimize import minimize


def hill_function(param, x):
    M, m, n, K = param
    return ((M-m)/(1 + power(x/K, n))) + m

# simulate data
ARA_MIN = 0
ARA_MAX = 100

class get_cascade():
    def __init__(self, K_araC):
        self.K_araC = K_araC

    def landscape(self, k_laci, k_tert, ara):
        param_TetR = [1.0, 0, 1, self.K_araC]
        TetR = hill_function(param_TetR, ara)
        param_laci = [10000, 1, 2.4, k_tert]
        LacI = hill_function(param_laci, TetR)
        param_out = [power(10, 1), power(10, 0), 0.7, k_laci]
        OUT = hill_function(param_out, LacI)
        return OUT

    def equation(self, vars):
        k_laci, k_tert = vars
        out = self.landscape(k_laci, k_tert, ara = ARA_MIN) -  self.landscape(k_laci, k_tert, ara = ARA_MAX)
        return -out

    def equation_c(self, k_laci, k_tert):
        out = self.landscape(k_laci, k_tert, ara = ARA_MIN) -  self.landscape(k_laci, k_tert, ara = ARA_MAX)
        return out


class Simulated(Dataset):
    """Simulated data with more complexe relationships
    """

    def __init__(self, nb_var, cor="exp", comp=False, alpha=45, center=[2.5, 2.5], temp=2):
        self.cor = cor
        self.alpha = (3.14/180) * alpha
        self.temp = temp
        if self.cor in ["exp", "tgaus", "cascade"]:
            self.center = center
        else:
            self.center = [2.5, 2.5]

        if comp:
            self.A = concatenate((uniform(1.0, 2, size=nb_var//2), uniform(4, 5, size=nb_var//2)))
            self.B = concatenate((uniform(1.0, 2, size=nb_var//2), uniform(4, 5, size=nb_var//2)))
        else:
            self.A = uniform(center[0]-2.5, center[0]+2.5, size=nb_var)
            self.B = uniform(center[1]-2.5, center[1]+2.5, size=nb_var)
        act_ai = arange(0, self.A.shape[0])
        act_bi = arange(0, self.B.shape[0])
        self.p1i, self.p2i = meshgrid(act_ai, act_bi)
        self.p1, self.p2, self.land = self.sim(self.A, self.B)
        data_np = array([self.p1i.flatten(), self.p2i.flatten(),
                         self.land.flatten() + normal(0, 0.05, self.land.flatten().shape)])
        self.data = tensor(data_np).transpose(0, 1)
        self.substitutions_tokens = [{i: i for i in range(nb_var)}, {i: i for i in range(nb_var)}]
        self.nb_val = 2 

    def sim(self, A, B):
        p1, p2 = meshgrid(A, B)
        if self.cor == "bio":
            land = self.mech_model(p1, p2)
        elif self.cor == "add":
            land = p1 + p2
        elif self.cor == "quad":
            land = p1 * p2
        elif self.cor == "comp":
            land = p1 + p2 - (p1 * p2)
        elif self.cor == "saddle":
            land = p1**2 - p2**2
        elif self.cor == "hat":
            land = sin(p1**2 + p2**2)
        elif self.cor == "exp":
            land = exp(-((self.center[0] - p1)**2 + (self.center[1] - p2)**2)/self.temp)
        elif self.cor == "tgaus":
            land = self.rotated_gaussian_mesh(p1, p2)
        elif self.cor == "cascade":
            land = self.cascade(p1, p2)
        land = 1*((land-land.mean())/land.std())
        # land = land
        # return p1, p2, land +  normal(0, 0.1, size=land.shape)
        return p1, p2, land

    def mech_model(self, A, B, omega=0.2756, neta=4.5514, fit_ben=3.6089,
                   toxic=0.0257, theta_A=0.0731, theta_B=0.0994, A_wt_1=1.5960,
                   B_wt_1=2.9926):
        "The model with parameters from Kemble et al 2020 Science advances"
        A += 0.1
        B += 0.1
        f_max =  1./neta
        flux = lambda A, B: 1. / (1./A + 1./B + neta)
        term_a = lambda A, B: (omega + (fit_ben * flux(A, B)) - (toxic/(f_max - flux(A, B))))
        term_b = lambda A, B: (1 - theta_A*A - theta_B*B)
        # making sure that the model stays at zero
        F = lambda A, B: (term_a(A, B) * term_b(A, B)) * \
            ((term_a(A, B) * term_b(A, B)) >= 0)
        return F(A, B) - F(A_wt_1, B_wt_1)

    def rotated_gaussian_mesh(self, X, Y):
        a = self.alpha
        m11 = cos(a)**2 / 2 + 2 * sin(a)**2
        m12_m21 = -sin(2*a) / 4 + sin(2*a)
        m22 = sin(a)**2 / 2 + 2 * cos(a)**2
        M = array([[m11, m12_m21],
                   [m12_m21, m22]])
        det_M = det(M)
        X = X - self.center[0]
        Y = Y - self.center[0]

        # Calculate the Gaussian function values for each (x, y) pair
        F = exp(-((X**2 * M[0,0] + 2 * X * Y * M[0,1] + Y**2 * M[1,1]) / (2 * npi * det_M)))
        return F

    def cascade(self, X, Y, K_araC = 1):
        initial_guess = [1,1]
        func_opt = get_cascade(K_araC=K_araC)
        # Find the maximum value of the equation
        result = minimize(func_opt.equation, initial_guess, method='Nelder-Mead')
        k_laci_max, k_tert_max =  result.x
        k_laci_xx, k_tert_yy = exp((X- self.center[0])*2)*k_laci_max, exp((Y - self.center[1])*2)*k_tert_max
        results = func_opt.equation_c(k_laci_xx, k_tert_yy)
        return results

    def plot(self, ax, fontsize=12):
        x_v = linspace(self.center[0]-2.5, self.center[0]+2.5, 400)
        y_v = linspace(self.center[1]-2.5, self.center[1]+2.5, 400)
        p1, p2, land = self.sim(x_v, y_v)
        ax.contourf(p1, p2, land, cmap="bwr", alpha=0.4)
        ax.set_xlabel("$\\phi_1$", fontsize=fontsize)
        ax.set_ylabel("$\\phi_2$", fontsize=fontsize)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
