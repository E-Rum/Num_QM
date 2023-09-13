import numpy as np


class Solver:
    def __init__(self, xmax, N, nnodes, pot_energy_func=None) -> None:
        self.xmax = xmax  # [0,xmax] -- domain of the solution
        self.N = N  # number of points on grid
        self.nnodes = nnodes  # number of nodes in the solution
        self.x = np.linspace(0, self.xmax, self.N)  # the computational grid
        self.delta_x = self.x[1] - self.x[0]  # step of our computational grid
        self.psi = np.zeros_like(self.x)  # initialize wave function
        self.pot_energy = (
            pot_energy_func(self.x) if pot_energy_func else self._get_pot_energy(self.x)
        )  # potential energy
        self.energy_range = {"min": min(self.pot_energy), "max": max(self.pot_energy)}
        self.energy = (self.energy_range["max"] + self.energy_range["min"]) / 2

    def _get_pot_energy(self, x):
        return x**2 / 2

    def _get_g(self):
        return 2 * (self.energy - self.pot_energy)

    def _get_f(self):
        return 1 + self.g * self.delta_x**2 / 12

    def _numerov_step(self, i_next, i, i_prev):
        return (
            (12 - 10 * self.f[i]) * self.psi[i] - self.f[i_prev] * self.psi[i_prev]
        ) / self.f[i_next]

    def _energy_bounds_update(self):
        if self._count_nodes() > self.nnodes:
            self.energy_range["max"] = self.energy
            self.energy = (self.energy_range["max"] + self.energy_range["min"]) / 2
        elif self._count_nodes() < self.nnodes:
            self.energy_range["min"] = self.energy
            self.energy = (self.energy_range["max"] + self.energy_range["min"]) / 2
        else:
            return True

    def _initialize_psi(self, flag):
        if flag == "outward":
            if self.nnodes % 2 == 0:
                self.psi[0] = 1
                self.psi[1] = (12 - 10 * self.f[0]) * self.psi[0] / 2 * self.f[1]
            else:
                self.psi[0] = 0
                self.psi[1] = 0.1
        elif flag == "inward":
            self.psi[-1] = self.delta_x
            self.psi[-2] = (12.0 - 10.0 * self.f[-1]) * self.psi[-1] / self.f[-2]
        else:
            raise ValueError("Unknown value for key")

    def _count_nodes(self):
        count = 0
        for j, k in zip(
            self.psi[: self.inversion_point - 1], self.psi[1 : self.inversion_point]
        ):
            if j * k <= 0:
                count += 1
        if self.nnodes % 2:
            return count * 2 - 1
        else:
            return count * 2

    def _solve_psi(self, key):
        if key == "outward":
            for i in range(1, self.inversion_point):
                self.psi[i + 1] = self._numerov_step(i + 1, i, i - 1)
        elif key == "inward":
            for i in range(len(self.x) - 2, self.inversion_point, -1):
                self.psi[i - 1] = self._numerov_step(i - 1, i, i + 1)
        else:
            raise ValueError("Unknown value for key")

    def _full_domain(self):
        if self.nnodes % 2:
            self.x = self.x = np.concatenate((np.flip(-self.x)[:-1], self.x))
            self.psi = np.concatenate((np.flip(-self.psi)[:-1], self.psi))
        else:
            self.x = self.x = np.concatenate((np.flip(-self.x)[:-1], self.x))
            self.psi = np.concatenate((np.flip(self.psi)[:-1], self.psi))

    def _solution_cycle(self, key):
        self.g = self._get_g()
        self.f = self._get_f()
        self.inversion_point = np.argmin(abs(self.pot_energy - self.energy))
        self._initialize_psi(key)
        self._solve_psi(key)
        return self.psi

    def _norm_wave_function(self):
        norm = np.sum(self.psi[1:] * self.psi[1:])
        norm = np.sqrt(self.delta_x * (self.psi[0] * self.psi[0] + 2.0 * norm))
        self.psi /= norm

    def find_energy(self):
        while self.energy_range["max"] - self.energy_range["min"] > 0.0001:
            key = "outward"
            self.psi = self._solution_cycle(key)
            if not self._energy_bounds_update():
                continue
            key = "inward"
            tmp = self.psi[self.inversion_point]
            self.psi = self._solution_cycle(key)
            self.psi[self.inversion_point :] *= tmp / self.psi[self.inversion_point]
            jump = (
                self.psi[self.inversion_point + 1]
                + self.psi[self.inversion_point - 1]
                - (14.0 - 12.0 * self.f[self.inversion_point])
                * self.psi[self.inversion_point]
            ) / self.delta_x
            if jump * self.psi[self.inversion_point] > 0:
                self.energy_range["max"] = self.energy
            else:
                self.energy_range["min"] = self.energy
            self.energy = (self.energy_range["max"] + self.energy_range["min"]) / 2
        self._norm_wave_function()
        self._full_domain()
        return self.energy, self.x, self.psi
