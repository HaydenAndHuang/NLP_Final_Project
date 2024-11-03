import os
import pandas as pd
from dotenv import load_dotenv
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaMA in-context baseline 

# Load environment variables from .env file
load_dotenv()

# Load LLaMA 3 model and tokenizer
model_name = "meta-llama/Meta-LLaMA-3-8B" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

def create_baseline_prompt(novice_caption, expert_examples, novice_examples):
    """Create a prompt for LLaMA using in-context learning examples."""
    prompt = "\n---\nGiven these examples below:\n\n"
    
    # Add examples for in-context learning
    for expert, novice in zip(expert_examples, novice_examples):
        prompt += f"Novice: {novice}\nExpert: {expert}\n\n"
        
    prompt += "Convert novice music descriptions to expert level descriptions.\n\n"
    prompt += f"Novice: {novice_caption}\nExpert:"
    
    return prompt

def generate_expert_description(novice_caption, expert_examples, novice_examples, max_length=512, temperature=0.7):
    """Generate an expert-level description using LLaMA 3."""
    instruction = (
        "You are a helpful assistant that converts novice-friendly music descriptions into expert descriptions.\n\n"
        "Transform the given input novice-level prompt into a prompt that a user with extensive music training and terminologies would use to prompt music generation models.\n\n"
        "Keep the instruments, genres, mood, and other information that represents the essence of the music.\n\n"
    )
    
    try:
        prompt = create_baseline_prompt(novice_caption, expert_examples, novice_examples)
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate output from the model
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
        
        # Decode the output to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = response.split("Expert:")[-1].strip()
        return result
    
    except Exception as e:
        print(f"Error generating description: {e}")
        return None

def main():
    # Read data from CSV file
    dataframe = pd.read_csv('musiccaps-updated.csv')
    
    # Read example pairs for in-context learning
    expert_novice_pair = pd.read_csv('expert_novice_captions.csv')
    expert_examples = expert_novice_pair['expert'].tolist()
    novice_examples = expert_novice_pair['novice'].tolist()
    
    # Initialize the expert descriptions list
    expert_descriptions = []
    batch_size = 10  # Adjust batch size based on your hardware capabilities

    # Batch processing
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Generating expert descriptions"):
        batch = dataframe['novice'][i:i + batch_size]
        
        for novice_caption in batch:
            expert_desc = generate_expert_description(novice_caption, expert_examples, novice_examples)
            expert_descriptions.append(expert_desc)
            time.sleep(1)  # Adjust or remove based on local processing needs

    # Add the expert descriptions to the dataframe
    dataframe.loc[:, 'gen_expert'] = expert_descriptions

    # Save the updated dataframe to a new CSV file
    dataframe.to_csv('musiccaps-baseline-expert.csv', index=False)
    print("Processing complete! Results saved to 'musiccaps-baseline-expert.csv'.")

if __name__ == "__main__":
    main()
