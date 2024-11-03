from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def create_prompt(expert_caption, expert_examples, novice_examples):
    """Create a prompt for GPT using in-context learning examples."""
    # New prompt instruction added
    prompt = "\n---\nGiven these examples below:\n\n"

    # Add examples for in-context learning
    for expert, novice in zip(expert_examples, novice_examples):
        prompt += f"Expert: {expert}\nNovice: {novice}\n\n"
        
    prompt += "Convert expert music descriptions to novice-friendly descriptions.\n\n"
    # Add the current caption to convert
    prompt += f"Expert: {expert_caption}\nNovice:"
    
    return prompt

def generate_novice_description(expert_caption, expert_examples, novice_examples):
    """Generate a novice-friendly description using GPT-3.5."""
    instruction = (
        "You are a helpful assistant that converts expert music descriptions into novice-friendly descriptions.\n\n"
        "Transform the given input expert-level prompt into a prompt that a user with little music experience would use to prompt music generation models.\n\n"
        "Keep the instruments, genres, mood, and other information that represents the essence of the music.\n\n"
        "Write the output succinctly in a coherent sentence.\n\n"
    )

    try:
        prompt = create_prompt(expert_caption, expert_examples, novice_examples)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating description: {e}")
        return None

def main():
    # Read data from CSV file
    df = pd.read_csv('musiccaps-public.csv')
    
    # Read example pairs for in-context learning
    expert_novice_pair = pd.read_csv('expert_novice_captions.csv')
    expert_examples = expert_novice_pair['expert'].tolist()
    novice_examples = expert_novice_pair['novice'].tolist()

    # Initialize the novice descriptions list
    novice_descriptions = []
    batch_size = 10

    # Batch processing
    for i in tqdm(range(0, len(df), batch_size), desc="Generating novice descriptions"):
        batch = df['caption'][i:i + batch_size]
        
        # Process each caption in the batch
        for expert_caption in batch:
            novice_desc = generate_novice_description(
                expert_caption=expert_caption,
                expert_examples=expert_examples,
                novice_examples=novice_examples
            )
            novice_descriptions.append(novice_desc)
            time.sleep(1)        

    # Add the novice descriptions to the dataframe
    # df['novice'] = novice_descriptions
    df.loc[:, 'novice'] = novice_descriptions

    # Save the updated dataframe to a new CSV file
    df.to_csv('musiccaps-novice8888.csv', index=False)
    print("Processing complete! Results saved to 'musiccaps-novice.csv'.")

if __name__ == "__main__":
    main()
