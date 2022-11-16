import math
from matplotlib import pyplot as plt
from copy import deepcopy


def log(x: float) -> float:
    """ Costum logarithm function """
    if x == 0:
        return float('-inf')
    return math.log(x)


def read_fasta_file(filename: str):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences


def get_test_params() -> tuple[list[list[float, ...], ...], list[dict, ...]]:
    letters = ['A', 'G', 'C', 'T']

    transition = [
        # NC  BC   C    EC   rBC  rC   rEC
        [0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],    # NC
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],    # BC
        [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0],    # C
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # EC
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],    # rBC
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],    # rC
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # rEC
    ]

    emission = [
        {l: 0.25 for l in letters},    # Non coding
        {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},     # Start codon
        {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},    # Coding
        {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},     # Stop codon
        {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},  # Reverse stop codon
        {l1 + l2 + l3: 1 / len(letters) ** 3 for l1 in letters for l2 in letters for l3 in letters},  # Reverse coding
        {l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters},  # Reverse start codon
    ]

    emission[1]['ATG'] = 1
    emission[3]['TAA'] = 1
    emission[4]['TTA'] = 1
    emission[6]['CAT'] = 1
    return transition, emission


def str_to_list(z: str) -> list[int, ...]:
    z_list = []
    n = 0
    while n < len(z):
        if z[n] == 'N':
            z_list.append(0)
            n += 1
        elif z[n] == 'C':
            if 'N' in z[n-3: n]:
                z_list.append(1)
            elif 'N' in z[n + 1: n + 4]:
                z_list.append(3)
            else:
                z_list.append(2)
            n += 3
        elif z[n] == 'R':
            if 'N' in z[n - 3: n]:
                z_list.append(4)
            elif 'N' in z[n + 1: n + 4]:
                z_list.append(6)
            else:
                z_list.append(5)
            n += 3
        else:
            raise NotImplementedError
    return z_list


def list_to_str(z: list[int, ...]) -> str:
    z_str = ''
    for value in z:
        if value == 0:
            z_str = z_str + 'N'
        elif value in [1, 2, 3]:
            z_str = z_str + 'CCC'
        elif value in [4, 5, 6]:
            z_str = z_str + 'RRR'
        else:
            raise NotImplementedError
    return z_str


def get_param_diff(transition1: list[list[float, ...], ...], emission1: list[dict, ...], transition2: list[list[float, ...], ...], emission2: list[dict, ...]) -> float:
    diff = 0.0
    for i in range(len(transition1)):
        for value1, value2 in zip(transition1[i], transition2[i], strict=True):
            diff += abs(value1 - value2)

    for i in range(len(emission1)):
        for value1, value2 in zip(emission1[i].values(), emission2[i].values(), strict=True):
            diff += abs(value1 - value2)

    return diff


def training_by_counting(K: int, letters: list[str, ...], d: dict, x: str, z: list[int, ...]) -> tuple[list[list[int, ...], ...], list[dict, ...]]:
    if isinstance(z, str):
        z = str_to_list(z)
    # % ------------------------ COMPUTE TRANSITION ------------------------ %
    transition = [[0 for _ in range(K)] for _ in range(K)]
    for z1, z2 in zip(z[:-1], z[1:]):   # Count
        transition[z1][z2] += 1

    for row in transition:  # Normalize
        S = sum(row)
        if S != 0:
            for i in range(len(row)):
                row[i] /= S

    # % ------------------------ COMPUTE EMISSION ------------------------ %
    emission = [{l: 0 for l in letters}] + \
               [{l1 + l2 + l3: 0 for l1 in letters for l2 in letters for l3 in letters}
                for _ in range(K - 1)]

    n = 0
    while n < len(z):  # Count
        emission[z[n]][x[n: n + d[z[n]]]] += 1
        n += d[z[n]]

    for i in range(K):  # Normalize
        S = sum(emission[i].values())
        if S != 0:
            for key in emission[i]:
                emission[i][key] /= S
    return transition, emission


def plot_solution(z1: str, z2: str):
    fig, ax = plt.subplots()
    y_pos = [0, 0.2]
    z1_plot = []
    z2_plot = []
    for z, z_plot in zip([z1, z2], [z1_plot, z2_plot]):
        for value in z:
            if value == 'N':
                z_plot.append('k')
            elif value == 'C':
                z_plot.append('b')
            elif value == 'R':
                z_plot.append('r')
            else:
                raise NotImplementedError

    for z_idx, z_plot in enumerate([z1_plot, z2_plot]):
        ax.scatter(list(range(len(z_plot))), len(z_plot) * [y_pos[z_idx]], c=z_plot)
    ax.set_ylim(-1.1, 1.1)
    plt.show()


def get_string_acc(s1: str, s2: str, prob: bool = True) -> float:
    diff = 0.0
    for a, b in zip(s1, s2, strict=True):
        if a == b:
            diff += 1.0
    if prob:
        diff /= len(s1)
    return diff


class HMM:
    def __init__(self, init_mode: str = 'test', x: None | str = None, z: None | str | list[int, ...] = None) -> None:
        self.d = {0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}     # How many values are emitted from each hidden state
        self.letters = ['A', 'G', 'C', 'T']                     # Bases
        self.K = 7
        self.initial = [1.0] + (self.K - 1) * [0.0]
        if init_mode == 'test':
            self.transition, self.emission = get_test_params()
        elif init_mode == 'from_counting':
            self.transition, self.emission = training_by_counting(K=self.K, letters=self.letters, d=self.d,
                                                                  x=x, z=z)
        else:
            raise NotImplementedError
        self.log_transition = deepcopy(self.transition)
        self.log_emission = deepcopy(self.emission)
        self.set_log_transition_emission()
        return

    def set_log_transition_emission(self) -> None:
        for i in range(len(self.log_transition[0])):
            for j in range(len(self.log_transition)):
                self.log_transition[i][j] = log(self.transition[i][j])

        for i in range(len(self.log_emission)):
            for key in self.log_emission[i].keys():
                self.log_emission[i][key] = log(self.emission[i][key])
        return

    def get_omega(self, x: str) -> list[list[float, ...], ...]:
        N = len(x)
        omega = [[float('-inf') for _ in range(N)] for _ in range(self.K)]     # Initializing the omega-matrix

        # % ------------------------ BASE CASE ------------------------ %
        # Setting the initial non-coding value
        n, k = 0, 0
        omega[k][n] = log(self.initial[k]) + self.log_emission[k][x[n]]

        # For n = 1, 2 the probability of remaining in the non-coding state is calculated
        k = 0
        for n in range(1, 3):
            omega[k][n] = omega[k][n - 1] + self.log_transition[k][k] + self.log_emission[k][x[n]]

        # The coding values are zero for n < 3 because they can only emit 3 letters
        for n in range(2):
            for k in range(1, self.K):
                omega[k][n] = float('-inf')

        # For n = 2 the system may start in the coding states by starting here
        n = 2
        for k in range(1, self.K):
            omega[k][n] = log(self.initial[k]) + self.log_emission[k][x[n - self.d[k] + 1: n + 1]]

        # % ------------------------ RECURSION ------------------------ %
        for n in range(3, N):
            for k in range(self.K):
                omega[k][n] = max([omega[kp][n - self.d[k]] + self.log_transition[kp][k] for kp in range(self.K)]) \
                              + self.log_emission[k][x[n - self.d[k] + 1: n + 1]]
        return omega

    def get_zstar(self, omega) -> list[int, ...]:
        N = len(omega[0])   # Number of data points
        zstar = []          # Container for optimal path

        # % ------------------------ BASE CASE ------------------------ %
        # Finding largest value in the last column of the omega-matrix.
        k_max, max_value = None, float('-inf')
        n = N - 1
        for k in range(self.K):
            if omega[k][n] > max_value:
                k_max, max_value = k, omega[k][n]
        zstar.append(k_max)     # Append this value to optimal path

        # % ------------------------ BACK TRACKING ------------------------ %
        n -= self.d[k_max]  # Starting point for back tracking. n will run from N - 1 to 0
        while n >= 0:   # When n is negative, we stop.
            # Find maximal value of omega[k][n] + log(transition[k][zstar]
            k_max, max_value = None, float('-inf')
            for k in range(self.K):
                value = omega[k][n] + self.log_transition[k][zstar[-1]] if n >= 0 else float('-inf')
                if value > max_value:
                    k_max, max_value = k, value
            zstar.append(k_max)  # Append this value to optimal path
            n -= self.d[k_max]  # Update n consistent with the chosen path
        return zstar[::-1]  # zstar was created in reverse direction (e.g. by adding the last element first).

    def Viterbi_training(self, x: str, ann: None | str = None, N_iter: int = 10 ** 2, tol: float = 1e-6) \
            -> tuple[list[float, ...], list[float, ...]] | list[float, ...]:
        n = 0
        param_diffs = [float('inf')]
        accs = []
        while n < N_iter and param_diffs[-1] > tol:
            omega = self.get_omega(x)
            zstar = self.get_zstar(omega)
            new_transition, new_emission = training_by_counting(K=self.K, letters=self.letters, d=self.d, x=x, z=zstar)
            param_diffs.append(get_param_diff(self.transition, self.emission, new_transition, new_emission))
            self.transition, self.emission = new_transition, new_emission
            if ann is not None:
                accs.append(get_string_acc(list_to_str(zstar), ann))
            n += 1
        if n == N_iter:
            print('% --------------- Maximum interations reached --------------- %')
        return param_diffs, accs


def profile_me():
    import numpy as np
    np.set_printoptions(linewidth=300)
    x = read_fasta_file('genome2.fa')['genome2']
    ann = read_fasta_file('true-ann2.fa')['true-ann2']
    hmm = HMM(init_mode='from_counting', x=x, z=ann)
    # hmm.Viterbi_training(x, ann, N_iter=1)
    omega = hmm.get_omega(x)
    zstar = hmm.get_zstar(omega)
    print(get_string_acc(list_to_str(zstar), ann))


def main():
    import cProfile
    import pstats
    from pstats import SortKey
    cProfile.run('profile_me()', 'restats')
    p = pstats.Stats('restats')
    p.sort_stats(SortKey.CUMULATIVE)
    p.print_stats()


if __name__ == '__main__':
    main()
