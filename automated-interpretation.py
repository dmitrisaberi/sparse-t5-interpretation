import openai
openai.api_key = "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"
context = "is there a rough, overarching theme tying most (i.e., >40%) of the following tokens together? We are using the term “theme” generously here — for example, if most of the outputs are numbers with no discernable pattern, then output “Yes: numbers”. This could also be a more broad one such as “religion”. There could also be two semantic meanings of a word that are both captured by the model — in this case, please provide both meanings.Please give your answer in the format 'Yes: (insert less than 4 word summary of concept)' or 'No', with no explanation."
def get_interpretability(act_prompts):
    interpretable_neurons = {}
    for neur in act_prompts.keys():
        for key in act_prompts[neur].keys():
            token_string = ": "
            for token in act_prompts[neur][key]:
                token_string = token_string + ", " + token
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context + token_string},
            ]
        )
        interpretation = response['choices'][0]['message']['content']
        if "Yes" in interpretation:
            interpretable_neurons[neur] = interpretation[4:]