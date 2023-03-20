import torch
from prompt_retrieval import getActivation


def topk_neurons(model, tokenizer, dataset, layer_number, k, encoder=True):
    # initialize top-k neurons dict
    top_k = {}
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
            input_ids = tokenizer.encode(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            target_ids = tokenizer.encode(target_text, padding=True, truncation=True, max_length=32, return_tensors="pt")

            # Generate the output from the model
            output_ids = model.generate(input_ids)

            # Compute top-k neurons after forward pass
            for key in activation.keys():
                if key not in top_k.keys():
                    top_k[key] = []
                for hidden_state in activation[key]:
                    # get top k neurons
                    assert hidden_state.shape[-1] == 3072
                    hidden_state = torch.squeeze(hidden_state, 0)
                    hidden_state = torch.sum(hidden_state, axis=0)
                    top_k[key] = top_k[key] + torch.argsort(hidden_state, dim=-1)[-k:].tolist()

            # Remove hooks (we can only use them once!)
            for handle in handles:
                handle.remove()
        
        return top_k

            # Decode the output and print the results
          #  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
          #  print("Input Text:", input_text)
          #  print("Target Text:", target_text)
          #  print("Generated Text:", output_text)
