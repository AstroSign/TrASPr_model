from keras.models import load_model
from pkg_resources import resource_filename
# from spliceai.utils import one_hot_encode
import numpy as np

output_dir = "./results/"
data_file = "/sample_data/dev.tsv"

context = 10000
# input_sequence = ['CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT',
                #   'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT']

def one_hot_encode(seq, strand='+'):
    IN_MAP = np.asarray([[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]

input_sequence = [[],[]]
hash_results = {}
with open(data_file) as f:
    next(f)
    for line in f:
        line = line.strip().split('\t')
        strand = line[1]
        seq_low = line[3]
        seq_high = line[4]

        if len(seq_low) < context:
            print(line[0])
        else:
            input_sequence[0].append([strand, seq_low])
            input_sequence[1].append([strand, seq_high])

x = [[],[]]
y = [None,None]
for i in range(len(input_sequence)):
    for strand, seq in input_sequence[i]:
        # x.append(one_hot_encode('N' + seq + 'N', strand))
        x[i].append(one_hot_encode(seq, strand))
    print("total cases:", len(x[i]))
    y[i] = np.mean([models[m].predict(np.array(x[i])) for m in range(5)], axis=0)
    print(f"shape:y[{i}]", y[i].shape)
y = np.concatenate(y, axis=1)
print("shape:y", y.shape)
# acceptor_prob = y[0, :, 1]
# donor_prob = y[0, :, 2]
np.save(output_dir + "pred_results.npy", y)
# import pdb; pdb.set_trace()

y = np.mean(y[:,:,1], axis=1)
print(y.shape)
