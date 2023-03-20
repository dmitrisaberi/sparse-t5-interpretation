import openai
openai.api_key = "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"
context = "is there a rough, overarching theme tying most (i.e., >40%) of the following tokens together? We are using the term “theme” generously here — for example, if most of the outputs are numbers with no discernable pattern, then output “Yes: numbers”. This could also be a more broad one such as “religion”. There could also be two semantic meanings of a word that are both captured by the model — in this case, please provide both meanings.Please give your answer in the format 'Yes: (insert less than 4 word summary of concept)' or 'No', with no explanation."
def get_interpretability(act_prompts):
    interpretable_neurons = {}
    for neur in act_prompts.keys():
        token_string = ": "
        for key in act_prompts[neur].keys():
            for token in act_prompts[neur][key]:
                token_string = token_string + ", " + token
        print(len(token_string.split()) + len(context.split()))
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context + token_string},
            ]
        )
        interpretation = response['choices'][0]['message']['content']
        print(interpretation)
        if "I'm sorry" in interpretation:
            print(token_string)
        if "Yes" in interpretation:
            print(interpretation[4:])
            interpretable_neurons[neur] = interpretation[4:]