import numpy as np
from tqdm import tqdm

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]


def build_vocab(vocab_path):
    vocab_file = open(vocab_path)
    vocab_data = vocab_file.readlines()
    vocab_file.close()

    index2word = {}
    word2index = {}
    word2index['PAD'] = 0
    index2word[0] = 'PAD'
    for idx, line in enumerate(vocab_data):
        word = line.rstrip()
        index2word[idx+1] = word
        word2index[word] = idx + 1
    index2word[len(vocab_data)+1] = 'EOS'
    word2index['EOS'] = len(vocab_data)+1
    return (word2index, index2word)

def build_data(data_path, word2index):
    data_file = open(data_path)
    data = data_file.readlines()
    data_file.close()
    input_seqs = []
    target_seqs = []
    for line in tqdm(data):
        pair = line.rstrip().split('|')
        input_ = []
        target_ = []
        for word in pair[0].split():
            input_.append(int(word))
        for word in pair[1].split():
            target_.append(int(word))
        target_.append(word2index['EOS'])
        input_seqs.append(input_)
        target_seqs.append(target_)
    return (input_seqs, target_seqs)

def batch_generator(input_seqs, target_seqs, batch_size, shuffle=True):
    data_size = len(input_seqs)
    num_batches = int((data_size-1)/batch_size) + 1

    input_lengths = np.array([len(seq) for seq in input_seqs])
    max_input_len = np.max(input_lengths)
    input_matrix = np.zeros([max_input_len, data_size], dtype=np.int32)
    target_lengths = np.array([len(seq) for seq in target_seqs])
    max_target_len = np.max(target_lengths)
    target_matrix = np.zeros([max_target_len, data_size], dtype=np.int32)
    target_lengths = np.array([len(seq) for seq in target_seqs])
    for i, seq in enumerate(input_seqs):
        for j, elem in enumerate(seq):
            input_matrix[j,i] = elem
    for i, seq in enumerate(target_seqs):
        for j, elem in enumerate(seq):
            target_matrix[j,i] = elem
    if shuffle:
        shuffle_indice = np.random.permutation(np.arange(data_size))
        input_shuffle = input_matrix[:,shuffle_indice]
        input_lengths_shuffle = input_lengths[shuffle_indice]
        target_shuffle = target_matrix[:,shuffle_indice]
        target_lengths_shuffle = target_lengths[shuffle_indice]
    else:
        input_shuffle = input_matrix
        input_lengths_shuffle = input_lengths
        target_shuffle = target_matrix
        target_lengths_shuffle = target_lengths
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, data_size)
        yield [input_shuffle[:,start_index:end_index], input_lengths_shuffle[start_index:end_index], target_shuffle[:,start_index:end_index], target_lengths_shuffle[start_index:end_index]]

def embed_fromGlove(glove_path, index2word):
    model = {}
    matrix = []
    print 'LOADING GLOVE'
    lines = open(glove_path).readlines()
    for line in lines:
        split_line = line.rstrip().split()
        word = split_line[0]
        embedding = [float(val) for val in split_line[1:]]
        model[word] = embedding
    word_num = len(index2word.keys())
    # random for 'POS'
    matrix.append(np.random.normal(-0.5 / 100, 0.5 / 100, size=100))
    # print word_num
    i = 0
    for idx in range(1,word_num - 1):
        if index2word[idx] in model:
            matrix.append(np.array(model[index2word[idx]]))
            i += 1
        else:
            matrix.append(np.random.normal(-0.5 / 100, 0.5 / 100, size=100))
    # for pad
    matrix.append(np.random.normal(-0.5 / 100, 0.5 / 100, size=100))
    f = open('embedding','w')
    matrix = np.array(matrix)
    # print i
    # print matrix.shape
    np.save(f, matrix)
    return matrix











