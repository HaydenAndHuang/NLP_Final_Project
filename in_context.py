import os
import pandas as pd
from dotenv import load_dotenv
import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaMA in-context baseline 

# Load environment variables from .env file
load_dotenv()

# Load LLaMA 3 model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Move the model to GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")

def create_baseline_prompt(novice_caption, expert_examples, novice_examples):
    """Create a prompt for LLaMA using in-context learning examples."""
    # Instruction to guide the model's behavior
    prompt = "\n---\nYou are a helpful assistant that converts novice-friendly music descriptions into expert descriptions. Given these examples below:\n\n"
       
    # Add examples for in-context learning
    for expert, novice in zip(expert_examples, novice_examples):
        prompt += f"Novice: {novice}\nExpert: {expert}\n\n"
    
    prompt += "\n---\nTransform the given input novice-level prompt into a prompt that a user with extensive music training and terminologies would use to prompt music generation models.\n\n"
    "Keep the instruments, genres, mood, and other information that represents the essence of the music.\n\n"

    prompt += f"Novice: {novice_caption}\nExpert:"
    
    return prompt

def generate_expert_description(novice_caption, expert_examples, novice_examples, max_new_tokens=256, temperature=0.7):
    """Generate an expert-level description using LLaMA 3.1."""
    try:
        prompt = create_baseline_prompt(novice_caption, expert_examples, novice_examples)
        
        # Tokenize the input prompt and move tensors to the GPU
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        # Generate output from the model
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Set pad token ID to avoid warnings
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
    batch_size = 10  # Adjust batch size 

    # Batch processing
    for i in tqdm(range(0, len(dataframe), batch_size), desc="Generating expert descriptions"):
        batch = dataframe['novice'][i:i + batch_size]
        
        for novice_caption in batch:
            expert_desc = generate_expert_description(novice_caption, expert_examples, novice_examples)
            expert_descriptions.append(expert_desc)
            time.sleep(1)  # Adjust or remove 

    # Add the expert descriptions to the dataframe
    dataframe.loc[:, 'gen_expert'] = expert_descriptions

    # Save the updated dataframe to a new CSV file
    dataframe.to_csv('musiccaps-baseline-expert-llama.csv', index=False)
    print("Processing complete! Results saved to 'musiccaps-baseline-expert-llama.csv'.")

if __name__ == "__main__":
    main()
