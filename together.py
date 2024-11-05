# -*- coding: utf-8 -*-
"""Together.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PkeeJl9KtAzL9EnxlKfdawE_Nd7cbfk0
"""

pip install together

import os
import pandas as pd
from together import Together
from tqdm import tqdm
os.environ["TOGETHER_API_KEY"] = "70f22d5bc0a1ac3c0eea5651f4d554f326a90a78559c62552935caef8dc8b013"

# Define the system message
dataframe = pd.read_csv('musiccaps-updated.csv')
expert_novice_pair = pd.read_csv('expert_novice_captions.csv')
expert_examples = expert_novice_pair['expert'].tolist()
novice_examples = expert_novice_pair['novice'].tolist()

instruction = "You are a helpful assistant that converts novice-friendly music descriptions into expert descriptions. "
instruction += "Given these examples below:\n\n"
for expert, novice in zip(expert_examples, novice_examples):
    instruction += f"Novice: {novice}\nExpert: {expert}\n\n"
instruction += "Transform the given input novice-level prompt into a prompt that a user with extensive music training and terminologies would write. "
instruction += "Keep the instruments, genres, mood, and other information that represents the essence of the music. "
instruction += "Use exactly 2 sentences."

novice_df = dataframe['novice'][4500:]
len(novice_df)

client = Together()


system_message = {
    "role": "system",
    "content": instruction
}

# Initialize an empty list to store the results
results = []

# Loop through each novice prompt and generate a response
for i, prompt in tqdm(enumerate(novice_df, 1), total=len(novice_df), desc="Processing prompts"):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            system_message,
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.25,
        top_p=0.5,
        top_k=50,
        repetition_penalty=0.9,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )

    #print(f"\nResponse for Prompt {i}:\n")
    # Collect the response content
    response_content = ""
    for token in response:
        if hasattr(token, 'choices'):
            response_content += token.choices[0].delta.content

    # Append the prompt and response as a dictionary to the results list
    results.append({"novice_prompt": prompt, "expert_prompt": response_content})

# Convert the results list to a DataFrame
output_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
output_df.to_csv("expert_prompts.csv", index=False)

print("Responses saved to expert_prompts.csv")

