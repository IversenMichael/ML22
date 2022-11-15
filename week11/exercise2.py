import math


def read_fasta_file(filename):
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


def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)



class HMM:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def translate_indices_to_path(indices):
    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])


def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]


def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]


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


def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0
               for i in range(len(true_ann))) / len(true_ann)


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
    g1 = read_fasta_file('C:\\Users\\au544901\\Documents\\GitHub\\ML22\\week11\\genome1.fa')
    ann1 = read_fasta_file('C:\\Users\\au544901\\Documents\\GitHub\\ML22\\week11\\true-ann1.fa')
    idx_stop = 100
    w = compute_w_log(hmm_7_state, g1['genome1'][:idx_stop])
    z = backtrack_log(hmm_7_state, g1['genome1'][:idx_stop], w)
    print(translate_indices_to_path(z))
    print(ann1['true-ann1'][:idx_stop])


if __name__ == '__main__':
    main()
