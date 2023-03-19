# NEURONS -> PROMPT CREATION
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
# Mapping neuron activations to prompts

# retrieve the activation layers using hooks
def getActivation(name, activation):
  # the hook signature
    def hook(model, input, output):
        if name not in activation.keys():
            activation[name] = [output.detach()]
        else:
            activation[name].append(output.detach())
    return hook

def get_activating_prompts(bucket_min, bucket_max, model, tokenizer, dataset, layer_number, encoder=True):
   # bucket_min = 100
   # bucket_max = 200
   # layer_number = 0
    import pickle
    with open('top-10-neurons.pkl', 'rb') as fp:
        top_k = pickle.load(fp)
    
    k = 10
    if encoder:
        layer = "encoder.block." + str(layer_number) + ".layer.1.DenseReluDense.act"
    else:
        layer = "decoder.block." + str(layer_number) + ".layer.2.DenseReluDense.act"

    counts, bins, bars = plt.hist(top_k[layer], bins=range(3072)) #, log=True)
    neurons = np.where(np.logical_and(counts >= bucket_min, counts <= bucket_max))[0]
    activating_prompts = {}
    for n in neurons:
        activating_prompts[n] = {}

    # Load the SQuAD dataset
    dataset = load_dataset("squad", split="validation")

    # Iterate over the dataset and feed it into the model
    with torch.no_grad():
        for example in dataset:
            # Get the input and target text from the example
            input_text = f"question: {example['question']} context: {example['context']}"
            target_text = example["answers"]["text"][0]

            # Register hooks
            handles = []
            activation = {}
            for name, module in model.named_modules():
                if "act" in name:
                    handles.append(module.register_forward_hook(getActivation(name, activation)))

            # Tokenize the input and target text
            input_ids = tokenizer.encode(input_text, truncation=True, max_length=512, return_tensors="pt")
            target_ids = tokenizer.encode(target_text, truncation=True, max_length=32, return_tensors="pt")

            # Get the tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Generate the output from the model
            output_ids = model.generate(input_ids)
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            

            for key in activation.keys():
                if key == layer:
                    tok = 0
                    for hidden_state in activation[key]:
                        # sanity check
                        assert hidden_state.shape[-1] == 3072
                        hidden_state = torch.squeeze(hidden_state, 0)
                        if encoder:
                            for token_idx in range(hidden_state.shape[0]):
                                activated_neurons = torch.nonzero(hidden_state[token_idx], as_tuple=True)[0]
                                for n in neurons:
                                    if n in activated_neurons:
                                        if input_text in activating_prompts[n].keys():
                                            activating_prompts[n][input_text].append(tokens[token_idx])
                                        else:
                                            activating_prompts[n][input_text] = [tokens[token_idx]]
                        else:
                            tok += 1
                            activated_neurons = torch.nonzero(hidden_state, as_tuple=True)[1]
                            for n in neurons:
                                if n in activated_neurons:
                                    if input_text in activating_prompts[n].keys():
                                        activating_prompts[n][input_text].append(output_tokens[tok])
                                    else:
                                        activating_prompts[n][input_text] = [output_tokens[tok]]
                            

            # Remove hooks (we can only use them once!)
            for handle in handles:
                handle.remove()
                
        return activating_prompts

            # Decode the output and print the results
          #  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
          #  print("Input Text:", input_text)
          #  print("Target Text:", target_text)
          #  print("Generated Text:", output_text)

