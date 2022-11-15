import math
from copy import copy


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)


class HMM:
    def __init__(self, init_probs, trans_probs, emission_probs):
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


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


def count_transitions_and_emissions(K: int, D: int, x_str: str, z_str: str):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    trans_count = make_table(K, K)
    emi_count = make_table(K, D)
    X = translate_observations_to_indices(x_str)
    Z = translate_path_to_indices(z_str)
    for z1, z2 in zip(Z[:-1], Z[1:]):
        trans_count[z1][z2] += 1
    for z, x in zip(Z, X):
        emi_count[z][x] += 1
    return trans_count, emi_count


def training_by_counting(K, D, x_str, z_str):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    X = translate_observations_to_indices(x_str)
    Z = translate_path_to_indices(z_str)

    init_probs = K * [0]
    for z in Z:
        init_probs[z] += 1 / len(Z)

    init_trans_counts = K * [0]
    for z in Z[:-1]:
        init_trans_counts[z] += 1

    trans_probs, emi_probs = count_transitions_and_emissions(K, D, x_str, z_str)
    for i in range(len(trans_probs)):
        for j in range(len(trans_probs[i])):
            trans_probs[i][j] /= init_trans_counts[i]

    init_emmi_count = copy(init_trans_counts)
    init_emmi_count[Z[-1]] += 1
    for i in range(len(emi_probs)):
        for j in range(len(emi_probs[i])):
            emi_probs[i][j] /= init_emmi_count[i]

    return HMM(init_probs, trans_probs, emi_probs)


def compute_w_log(model, x_str: str):
    x = translate_observations_to_indices(x_str)
    k = len(model.init_probs)
    n = len(x)
    w = make_table(k, n)

    for i in range(k):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])
    for j in range(1, n):  # Target z-state
        for i in range(k):  # Loop down each column of the omega-matrix
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
            # print(f'{z_star[-1]=}', f'{x[j]=}')

            value = log(model.emission_probs[z_star[-1]][x[j + 1]]) + w[i][j] + log(model.trans_probs[i][z_star[-1]])
            # print(f'{value=}')
            if value > max_value:
                max_value = value
                argmax = i
        z_star.append(argmax)

    return z_star[::-1]


def viterbi_update_model(model: HMM, x_str: str):
    """
    return a new model that corresponds to one round of Viterbi training,
    i.e. a model where the parameters reflect training by counting on x
    and z_vit, where z_vit is the Viterbi decoding of x under the given
    model.
    """
    K, D = 7, 4
    w = compute_w_log(model, x_str)
    z_star = backtrack_log(model, x_str, w)
    return training_by_counting(K, D, x_str, z_star)


def get_model_diff(model1: HMM, model2: HMM) -> float:
    diff = 0.0

    for i in range(len(model1.init_probs)):
        diff += abs(model1.init_probs[i] - model2.init_probs[i])

    for i in range(len(model1.trans_probs)):
        for j in range(len(model1.trans_probs[i])):
            diff += abs(model1.trans_probs[i][j] - model2.trans_probs[i][j])

    for i in range(len(model1.emission_probs)):
        for j in range(len(model1.emission_probs[i])):
            diff += abs(model1.emission_probs[i][j] - model2.emission_probs[i][j])

    return diff


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


def main():
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
    K, D = 7, 4
    x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
    z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'
    hmm_7_state_tbc = training_by_counting(7, 4, x_long, z_long)

    w = compute_w_log(hmm_7_state, x_long)
    z_vit = backtrack_log(hmm_7_state, x_long, w)

    w_tbc = compute_w_log(hmm_7_state_tbc, x_long)
    z_vit_tbc = backtrack_log(hmm_7_state_tbc, x_long, w_tbc)

    for _ in range(10):
        hmm_7_state0 = hmm_7_state
        hmm_7_state = viterbi_update_model(hmm_7_state0, x_long)
        print(get_model_diff(hmm_7_state0, hmm_7_state))


if __name__ == '__main__':
    main()
