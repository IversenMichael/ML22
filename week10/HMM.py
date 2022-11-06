import math


class HMM:
    def __init__(self, init_probs: list[float, ...], trans_probs: list[list[float, ...], ...],
                 emission_probs: list[list[float, ...], ...]):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))


def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])


def validate_hmm(model: HMM):
    if abs(sum(model.init_probs) - 1) > 1e-12:
        return False
    for row in model.trans_probs + model.emission_probs:
        if abs(sum(row) - 1) > 1e-12:
            return False
    for f in model.init_probs:

        if not 0 <= f <= 1:
            return False
    for row in model.trans_probs + model.emission_probs:
        for f in row:
            if not 0 <= f <= 1:
                return False
    return True


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


def joint_prob(model: HMM, x_str: str, z_str: str):
    x = translate_observations_to_indices(x_str)
    z = translate_path_to_indices(z_str)
    prob = model.init_probs[z[0]]
    for i, j in zip(z[:-1], z[1:]):
        prob *= model.trans_probs[i][j]
    for i, j in zip(z, x):
        prob *= model.emission_probs[i][j]
    return prob


def joint_prob_log(model, x_str, z_str):
    x = translate_observations_to_indices(x_str)
    z = translate_path_to_indices(z_str)
    log_prob = log(model.init_probs[z[0]])
    for i, j in zip(z[:-1], z[1:]):
        log_prob += log(model.trans_probs[i][j])
    for i, j in zip(z, x):
        log_prob += log(model.emission_probs[i][j])
    return log_prob


def first_part():
    init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
    trans_probs_7_state = [
        [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
    ]
    emission_probs_7_state = [
        #   A     C     G     T
        [0.30, 0.25, 0.25, 0.20],
        [0.20, 0.35, 0.15, 0.30],
        [0.40, 0.15, 0.20, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.40, 0.30, 0.10],
        [0.30, 0.20, 0.30, 0.20],
        [0.15, 0.30, 0.20, 0.35],
    ]
    hmm_7_state = HMM(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)
    print(validate_hmm(hmm_7_state))

    x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
    z_short = '33333333333321021021021021021021021021021021021021'

    print(joint_prob(hmm_7_state, x_short, z_short))
    print(joint_prob_log(hmm_7_state, x_short, z_short))

    x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
    z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'
    for i in range(100, len(x_long), 100):
        if joint_prob(hmm_7_state, x_long[:i], z_long[:i]) == 0:
            print(i)
            break
    print(joint_prob(hmm_7_state, x_long[:i], z_long[:i]))
    print(joint_prob_log(hmm_7_state, x_long[:i], z_long[:i]))


def make_table(m, n) -> list[list[int, ...], ...]:
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


def compute_w(model: HMM, x_str: str) -> list[list[int, ...], ...]:
    x = translate_observations_to_indices(x_str)
    k = len(model.init_probs)
    n = len(x)
    w = make_table(k, n)

    for i in range(k):
        w[i][0] = model.init_probs[i] * model.emission_probs[i][x[0]]
    for j in range(1, n):  # Loop down each column of the omega-matrix
        for i in range(k):  # Target z-state
            w_max = 0
            for a in range(k):  # Index for finding max of function
                w_max = max(w_max, w[a][j-1] * model.trans_probs[a][i])
            w[i][j] = w_max * model.emission_probs[i][x[j]]
    return w


def opt_path_prob(w) -> tuple[int, float]:
    argmax = 0
    max_value = 0
    for i in range(len(w)):
        if w[i][-1] > max_value:
            argmax = i
            max_value = w[i][-1]
    return argmax, max_value


def backtrack(model: HMM, x_str: str, w: list[list[int, ...], ...]):
    x = translate_observations_to_indices(x_str)
    z_star = []

    max_value = -1
    argmax = None
    for i in range(len(w)):
        if w[i][-1] > max_value:
            max_value = w[i][-1]
            argmax = i
    z_star.append(argmax)

    for j in range(0, len(w[0]) - 1)[::-1]:
        max_value = -1
        argmax = None
        for i in range(len(w)):
            value = model.emission_probs[z_star[-1]][x[j]] * w[i][j] * model.trans_probs[i][z_star[-1]]
            if value > max_value:
                max_value = value
                argmax = i
        z_star.append(argmax)

    return z_star[::-1]


def compute_w_log(model, x_str: str):
    x = translate_observations_to_indices(x_str)
    k = len(model.init_probs)
    n = len(x)
    w = make_table(k, n)

    for i in range(k):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])
    for j in range(1, n):  # Loop down each column of the omega-matrix
        for i in range(k):  # Target z-state
            w_max = float('-inf')
            for a in range(k):  # Index for finding max of function
                w_max = max(w_max, w[a][j-1] + log(model.trans_probs[a][i]))
            w[i][j] = w_max + log(model.emission_probs[i][x[j]])
    return w


def opt_path_prob_log(w) -> tuple[int, float]:
    argmax = 0
    max_value = float('-inf')
    for i in range(len(w)):
        if w[i][-1] > max_value:
            argmax = i
            max_value = w[i][-1]
    return argmax, max_value


def backtrack_log(model: HMM, x_str: str, w: list[list[int, ...], ...]):
    x = translate_observations_to_indices(x_str)
    z_star = []

    max_value = float('-inf')
    argmax = None
    for i in range(len(w)):
        if w[i][-1] > max_value:
            max_value = w[i][-1]
            argmax = i
    z_star.append(argmax)

    for j in range(0, len(w[0]) - 1)[::-1]:
        max_value = float('-inf')
        argmax = None
        for i in range(len(w)):
            value = log(model.emission_probs[z_star[-1]][x[j]]) + w[i][j] + log(model.trans_probs[i][z_star[-1]])
            if value > max_value:
                max_value = value
                argmax = i
        z_star.append(argmax)

    return z_star[::-1]


def Viterbi():
    init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
    trans_probs_7_state = [
        [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
    ]
    emission_probs_7_state = [
        #   A     C     G     T
        [0.30, 0.25, 0.25, 0.20],
        [0.20, 0.35, 0.15, 0.30],
        [0.40, 0.15, 0.20, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.40, 0.30, 0.10],
        [0.30, 0.20, 0.30, 0.20],
        [0.15, 0.30, 0.20, 0.35],
    ]
    hmm_7_state = HMM(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)
    x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
    z_short = '33333333333321021021021021021021021021021021021021'
    x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
    z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'
    w_log = compute_w_log(hmm_7_state, x_long)
    w = compute_w(hmm_7_state, x_long)
    z_viterbi_log = backtrack_log(hmm_7_state, x_long, w_log)
    z_viterbi = backtrack(hmm_7_state, x_long, w)
    print(z_viterbi)
    print(z_viterbi_log)

if __name__ == '__main__':
    Viterbi()
