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
        initial = [0.25, 0.25, 0.25, 0.25]

        transition = [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [1.0, 0.0, 0.0, 0.0]
        ]

        letters = ['A', 'G', 'C', 'T']
        emmision = [
            {l: 0.25 for l in letters},
            {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},
            {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},
            {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},
        ]
        return initial, transition, emmision

    def _get_omega(self, x: str) -> list[list[float, ...], ...]:
        N = len(x)
        omega = [[0.0 for _ in range(N)] for _ in range(self.K)]     # Initializing the omega-matrix

        # % ------------------------ BASE CASE ------------------------ %
        n, k = 0, 0
        omega[k][n] = self.initial[k] * self.emission[k][x[n]]
        for n in range(1, 3):
            omega[k][n] = omega[k][n - 1] * self.transition[k][k] * self.emission[k][x[n]]

        n = 2
        for k in range(1, 4):
            print(n - self.d[k], n)
            omega[k][n] = self.initial[k] * self.emission[k][x[n - self.d[k] + 1: n + 1]]

        # % ------------------------ RECURSION ------------------------ %
        n = 1

        return omega


def main():
    hmm = HMM()
    x = 'AAA'
    print(hmm._get_omega(x))
    print(hmm.emission[1])

if __name__ == '__main__':
    main()
