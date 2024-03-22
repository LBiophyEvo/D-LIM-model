from torch.utils.data import Dataset
from torch import tensor, rand, cdist, randn, exp as texp
from numpy.random import normal, uniform
from numpy import linspace, arange, meshgrid, array, exp, sin, concatenate


class Simulated(Dataset):
    """Simulated data with more complexe relationships
    """

    def __init__(self, nb_var, cor="exp", comp=False):
        self.cor = cor
        if not comp:
            self.A = uniform(0.0, 5, size=nb_var)
            self.B = uniform(0.0, 5, size=nb_var)
        else:
            self.A = concatenate((uniform(1.0, 2, size=nb_var//2), uniform(4, 5, size=nb_var//2)))
            self.B = concatenate((uniform(1.0, 2, size=nb_var//2), uniform(4, 5, size=nb_var//2)))
        self.p1, self.p2, self.p1i, self.p2i, self.land = self.sim(self.A, self.B)
        data_np = array([self.p1i.flatten(), self.p2i.flatten(),
                         self.land.flatten() + normal(0, 0.1, self.land.flatten().shape)])
        self.data = tensor(data_np).transpose(0, 1)

    def sim(self, A, B):
        act_ai = arange(0, A.shape[0])
        act_bi = arange(0, B.shape[0])

        p1, p2 = meshgrid(A, B)
        p1i, p2i = meshgrid(act_ai, act_bi)
        if self.cor == "bio":
            land = 10*self.mech_model(p1, p2)
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
            ref_1x, ref_1y = 2., 2.
            land = 10 * (exp(-((ref_1x - p1)**2 + (ref_1y - p2)**2)))
        return p1, p2, p1i, p2i, land

    def mech_model(self, A, B, omega=0.2756, neta=4.5514, fit_ben=3.6089,
                   toxic=0.0257, theta_A=0.0731, theta_B=0.0994, A_wt_1=1.5960,
                   B_wt_1=2.9926):
        "The model with parameters from Kemble et al 2020 Science advances"
        f_max =  1./neta
        flux = lambda A, B: 1. / (1./A + 1./B + neta)
        term_a = lambda A, B: (omega + (fit_ben * flux(A, B)) - (toxic/(f_max - flux(A, B))))
        term_b = lambda A, B: (1 - theta_A*A - theta_B*B)
        # making sure that the model stays at zero
        F = lambda A, B: (term_a(A, B) * term_b(A, B) + normal(0, 0.1)) * \
            ((term_a(A, B) * term_b(A, B)) >= 0)
        return F(A, B) - F(A_wt_1, B_wt_1)

    def plot(self, ax):
        x_v = linspace(0, 5, 200)
        y_v = linspace(0, 5, 200)
        p1, p2, _, __, land = self.sim(x_v, y_v)
        ax.contourf(p1, p2, land, cmap="bwr", alpha=0.4)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
