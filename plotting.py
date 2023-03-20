import matplotlib.pyplot as plt
import pickle

def plot_topk(layer_number, encoder=True):
    with open("top-10-neurons.pkl", 'rb') as fp:
        top_k = pickle.load(fp)
    if encoder:
        layer = "encoder.block." + str(layer_number) + ".layer.1.DenseReluDense.act"
    else:
        layer = "encoder.block." + str(layer_number) + ".layer.2.DenseReluDense.act"
    values = top_k[layer]
    plt.figure(figsize=(10,6))

    counts, bins, bars = plt.hist(values, bins=range(3072)) #, log=True)
    plt.title("Top-" + str(k) + " neuron presence; Encoder layer " + str(layer_number))
    plt.show()