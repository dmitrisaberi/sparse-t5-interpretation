import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_topk(layer_number, k=10, encoder=True, log_scale=False):
    with open("top-10-neurons.pkl", 'rb') as fp:
        top_k = pickle.load(fp)
    if encoder:
        layer = "encoder.block." + str(layer_number) + ".layer.1.DenseReluDense.act"
    else:
        layer = "decoder.block." + str(layer_number) + ".layer.2.DenseReluDense.act"
    values = top_k[layer]
    plt.figure(figsize=(10,6))

    counts, bins, bars = plt.hist(values, bins=range(3072), log=log_scale)
    plt.title("Top-" + str(k) + " neuron firing count; Decoder layer " + str(layer_number))
    plt.show()
    return counts

def plot_layer_sparsities():
    with open("squad_layer_sparsity.pkl", "rb") as fp:
        sparsities = pickle.load(fp)
    avg_sparsities = {}
    for key in sparsities.keys():
        s = np.array(sparsities[key])
        avg_sparsities[key] = np.mean(s)   
#    names = list(avg_sparsities.keys())
    values = list(avg_sparsities.values())
    shortened_names = []
    for i in range(24):
        if i < 12:
            shortened_names.append("e" + str(i))
        else:
            shortened_names.append("e" + str(i%12))
    plt.figure(figsize=(10,6))
    plt.bar(range(len(avg_sparsities)), values, tick_label=shortened_names)
    plt.title("Average percentage nonzeros of activation layers")
    plt.show()