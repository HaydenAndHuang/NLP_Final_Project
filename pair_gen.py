from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to get completion from OpenAI API
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Degree of randomness of the model's output
    )
    return response.choices[0].message.content

# Function to generate a prompt for the OpenAI API
def generate_prompt(expert_0, expert_1, expert_2, expert_3, expert_4, novice_0, novice_1, novice_2, novice_3, novice_4, caption):
    prompt = f"""
        
        ---
        Given these examples below:
        <expert>: {expert_0}
        <novice>: {novice_0}

        <expert>: {expert_1}
        <novice>: {novice_1}

        <expert>: {expert_2}
        <novice>: {novice_2}
        
        <expert>: {expert_3}
        <novice>: {novice_3}

        <expert>: {expert_4}
        <novice>: {novice_4}
        ---

        Transform the given input expert-level prompt into a prompt that a user with little music experience would use to prompt music generation models. 

        Keep the instruments, genres, mood, and other information that represents the essence of the music.
    
        Write the output succinctly in a coherent sentence.

        <expert>: {caption}
        <novice>:
        """
    return prompt

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Read data from CSV file
df = pd.read_csv('musiccaps-public.csv')

expert_0 = df['caption'][0] # Christian
expert_1 = df['caption'][15] # Electronic music
expert_2 = df['caption'][19] # Gospel
expert_3 = df['caption'][30] # rock
expert_4 = df['caption'][41] # classical
novice_0 = "A melancholic piano song with a female singer that would be played at church"
novice_1 = "R&B, male singer, string, strong bass, drums, suited for an intimate setting"
novice_2 = "Gospel music for children, bass and drums, spiritual feeling"
novice_3 = "Rock music with guitar and drums, with angry and aggressive vocals"
novice_4 = "Calming classical music similar to Bach with harp"

# Generate novice prompts for each expert caption and add as a new column
prompt = generate_prompt(expert_0, expert_1, expert_2, expert_3, expert_4, novice_0, novice_1, novice_2, novice_3, novice_4, caption)
df['novice'] = df['caption'].apply(lambda caption: get_completion(prompt))

# Save the updated DataFrame to a new CSV
df.to_csv('musiccaps-with-novice.csv', index=False)

# Optional: Print the result
print(df.head())