import openai
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

openai.api_key = "API-KEY-HERE"
context = "is there a rough, overarching theme tying most (i.e., >40%) of the following tokens together? We are using the term “theme” generously here — for example, if most of the outputs are numbers with no discernable pattern, then output “Yes: numbers”. This could also be a more broad one such as “religion”. Please give your answer in the format 'Yes: (insert less than 4 word summary of concept)' or 'No', with no explanation."
def get_interpretability(act_prompts):
    interpretable_neurons = {}
    for neur in act_prompts.keys():
        flag = False
        token_string = ": "
        for key in act_prompts[neur].keys():
            for token in act_prompts[neur][key]:
                token_string = token_string + ", " + token
                if len(encoding.encode(context + token_string)) > 4050:
                    flag = True
                    break
            if flag:
                break
        print(len(encoding.encode(context + token_string)))
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context + token_string},
            ]
        )
        interpretation = response['choices'][0]['message']['content']
        print(token_string)
        print(interpretation)
       # if "I'm sorry" in interpretation:
           # print(token_string)
        if "Yes" in interpretation:
           # print(interpretation[4:])
            interpretable_neurons[neur] = interpretation[4:]

    return interpretable_neurons

def fraction_interpretable(interpretable_neurons):
    interpretable = 0
    total = 0
    for neur in interpretable_neurons.keys():
        total += 1
        if "Yes" in interpretable_neurons[neur]:
            interpretable += 1
    return interpretable/total