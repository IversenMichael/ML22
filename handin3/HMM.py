import math


def log(x: float) -> float:
    if x == 0:
        return float('-inf')
    return math.log(x)


class HMM:
    def __init__(self, initial: list[float, ...] | None = None, transition: list[list[float, ...], ...] | None = None,
                 emission: list[dict, ...] | None = None) -> None:
        if initial is None and transition is None and emission is None:
            self.initial, self.transition, self.emission = self._get_test_params()
        else:
            self.initial = initial
            self.transition = transition
            self.emission = emission

        self.K = len(self.initial)
        self.d = {0: 1, 1: 3, 2: 3, 3: 3}
        return

    @staticmethod
    def _get_test_params() -> tuple[list[float], list[list[float, ...], ...], list[dict, ...]]:
        initial = [1, 0, 0, 0]

        transition = [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [1.0, 0.0, 0.0, 0.0]
        ]

        letters = ['A', 'G', 'C', 'T']
        emmision = [
            {l: 0.25 for l in letters},
            {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},
            {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},
            {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},
        ]
        emmision[1]['ATG'] = 1
        emmision[3]['TAT'] = 1
        return initial, transition, emmision

    def get_omega(self, x: str) -> list[list[float, ...], ...]:
        N = len(x)
        omega = [[float('-inf') for _ in range(N)] for _ in range(self.K)]     # Initializing the omega-matrix

        # % ------------------------ BASE CASE ------------------------ %
        # Setting the initial non-coding value
        n, k = 0, 0
        omega[k][n] = log(self.initial[k]) + log(self.emission[k][x[n]])

        # For n = 1, 2 the probability of remaining in the non-coding state is calculated
        k = 0
        for n in range(1, 3):
            omega[k][n] = omega[k][n - 1] + log(self.transition[k][k]) + log(self.emission[k][x[n]])

        # The coding values are zero for n < 3 because they can only emit 3 letters
        for n in range(2):
            for k in range(1, 4):
                omega[k][n] = float('-inf')

        # For n = 2 the system may start in the coding states by starting here
        n = 2
        for k in range(1, 4):
            omega[k][n] = log(self.initial[k]) + log(self.emission[k][x[n - self.d[k] + 1: n + 1]])

        # % ------------------------ RECURSION ------------------------ %
        for n in range(3, N):
            for k in range(4):
                max_value = float('-inf')
                for kp in range(4):
                    max_value = max(max_value, omega[kp][n - self.d[k]] + log(self.transition[kp][k]))
                omega[k][n] = max_value + log(self.emission[k][x[n - self.d[k] + 1: n + 1]])
        return omega

    def get_zstar(self, omega):
        N = len(omega[0])
        zstar = []

        # % ------------------------ BASE CASE ------------------------ %
        k_max, max_value = None, float('-inf')
        for k in range(4):
            if omega[k][-1] > max_value:
                k_max, max_value = k, omega[k][-1]
        zstar.append(k_max)

        # % ------------------------ BACK TRACKING ------------------------ %
        n = N - 1
        while n != 0:
            k_max, max_value = None, float('-inf')
            for k in range(4):
                if k == 0:
                    np = n - 1
                else:
                    np = n - 3
                if np < 0:
                    value = float('-inf')
                else:
                    value = omega[k][np] + log(self.transition[k][zstar[-1]])
                if value > max_value:
                    k_max, max_value = k, value
            zstar.append(k_max)
            if k_max == 0:
                n = n - 1
            else:
                n = n - 3
            if n < 0:
                raise NotImplementedError
        return zstar[::-1]


def main():
    import numpy as np
    np.set_printoptions(linewidth=300)
    hmm = HMM()
    x = ''
    omega = hmm.get_omega(x)
    for row in omega:
        print(np.round(row, 1))
    print(hmm.get_zstar(omega))


if __name__ == '__main__':
    main()
