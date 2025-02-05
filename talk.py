from transformers import pipeline
from transformers import AutoTokenizer
import random

f = open("./users_active.txt", "r", encoding="utf-8")
users_active = f.read().splitlines()
f2 = open("./users.txt", "r", encoding="utf-8")
users = f2.read().splitlines()
def anonymize_users(users):
    users_map = {}
    counter = 0
    for name in users:
        users_map[name] = name[0] + f"User_{counter}" + name[-2:]
        counter += 1
    return users_map
users_map = anonymize_users(users)


generator = pipeline("text-generation", model="papuGaPT2-finetune/checkpoint-149526", tokenizer="flax-community/papuGaPT2")
generator_original = pipeline("text-generation", model="flax-community/papuGaPT2", tokenizer="flax-community/papuGaPT2")
generator_small = pipeline("text-generation", model="papuga-finetune-2/checkpoint-52500", tokenizer="flax-community/papuGaPT2")

# generator = pipeline("text-generation", model="scratch-model-gpt2/checkpoint-177565", tokenizer="gpt2")
# generator_original = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# generator = pipeline("text-generation", model="distilgpt2-finetune/checkpoint-224289", tokenizer="distilgpt2")
# generator_original = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("flax-community/papuGaPT2")
me =  "⊙imie nazwisko⊙:"
for i in range(60):
    print("write something")
    prompt = input()
    prompt = me + prompt + '\n'
    prompt_original = prompt
    prompt_small = prompt
    user = random.choice(users_active)
    user = me
    
    output = generator(prompt + user, max_new_tokens=32, num_return_sequences=1, pad_token_id=50256)
    response = output[0]['generated_text'].split(prompt,1)[1].split('\n')[0]
    
    output_original = generator_original(prompt_original + user, max_new_tokens=32, num_return_sequences=1, pad_token_id=50256)
    response_original = output_original[0]['generated_text'].split(prompt_original,1)[1].split('\n')[0]
    
    output_small = generator_small(prompt_small + user, max_new_tokens=32, num_return_sequences=1, pad_token_id=50256)
    response_small = output_small[0]['generated_text'].split(prompt_small,1)[1].split('\n')[0]
    
    print(f"Response: {response.replace(user, users_map[user])}")
    print(f"Response original: {response_original.replace(user, users_map[user])}")
    print(f"Response_small: {response_small.replace(user, users_map[user])}")
    prompt += response + '\n'
    prompt_original += response_original + '\n'
    response_small += response_small + '\n'
    print("-" * 50)
