# NLP_Final_Project
---

# create virtual env 
Run in terminal
### create virtual env with 
```
conda env export > environment.yml
```

### putting your key in a file called `.env`
```
echo "OPENAI_API_KEY=xxxxxxx" >> .env
```

# Data

**musiccaps-public.csv**  
Original dataset.

**expert_novice_captions.csv**  
This dataset contains 24 handwritten expert-novice examples.

**musiccaps-updated.csv**  
This dataset contains a novice column generated by GPT-3.5.



# pair_gen.ipynb
和锴锴发到群上的 code 一样，set up the visual env first and run the code. 

# pair_gen.py (ready to use)
This script is designed to transform expert-level music descriptions (captions) into novice-level prompts using the OpenAI API. Here’s how it works:
1. **Input :**
	• The code reads a CSV file named **musiccaps-public.csv**, which contains expert-level music descriptions (captions).
2. **Output:**
	• The updated DataFrame, which now includes both expert captions and their novice versions, is saved to a new CSV file named **musiccaps-with-novice.csv**.

**Input path:**
```
# Read data from CSV file

df = pd.read_csv('musiccaps-public.csv')
```

**Output:**
The output files will be saved to the same path as the code path.
```
# Save the updated DataFrame to a new CSV

df.to_csv('musiccaps-with-novice.csv', index=False)
```
  
**How to Run the Code**:
```
python pair_gen.py
```


# together.py
This code takes novice-level music descriptions from a CSV file, transforms them into expert-level descriptions using a model, and saves the output to a new CSV file.

### Inputs
- `musiccaps-updated.csv`: Contains novice-level music descriptions, with the code processing entries from row 4500 onward.
- `expert_novice_captions.csv`: Provides pairs of novice and expert descriptions, which are used to create the instruction message for the model.

### Outputs
- `expert_prompts.csv`: A CSV file containing the novice prompts from `musiccaps-updated.csv` alongside their transformed expert-level descriptions.

### Model
The code uses the `"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"` model from the `Together` API to generate expert-level descriptions based on novice inputs.

### Important Parameters
- **max_tokens**: Set to 60, limiting the maximum tokens in the model's response.
- **temperature**: Set to 0.25, controlling response randomness. A lower value produces more focused outputs.
- **top_p**: Set to 0.5, controlling nucleus sampling, so the model samples from tokens within the top 50% probability.
- **top_k**: Set to 50, considering the top 50 tokens for each step in the generation.
- **repetition_penalty**: Set to 0.9, discouraging repeated tokens.
- **stop tokens**: Specifies `"<|eot_id|>"` and `"<|eom_id|>"` as stop tokens, ending generation when encountered.

### Summary
The code fine-tunes novice prompts to make them sound more expert, keeping the same musical essence, style, and structure, then stores the results. This transformation aims to produce polished, professional-level music descriptions.



# in_context.py
This code is designed to use the **LLaMA 3.1 model** to transform novice-level music descriptions into expert-level descriptions using in-context learning. Here’s a breakdown of its main components, inputs, outputs, model, and important parameters:

### 1. **Purpose and Functionality**
   - The code takes novice descriptions of music and converts them into expert-level descriptions, which would appeal to or be generated by users with extensive music knowledge.
   - It uses **in-context learning** to provide the model with example pairs of novice and expert descriptions, helping it understand the transformation.

### 2. **Input and Output**
   - **Input:**
      - `novice_caption`: A novice-level music description to be converted.
      - `expert_examples` and `novice_examples`: Lists of expert and novice example pairs used to guide the model's behavior.
      - `musiccaps-updated.csv`: A CSV file with novice descriptions that the model will convert.
      - `expert_novice_captions.csv`: A CSV file with example pairs of expert and novice descriptions for in-context learning.

   - **Output:**
      - The transformed expert-level descriptions are saved in `musiccaps-baseline-expert-llama.csv`, with a new column named `gen_expert` containing the model's outputs.

### 3. **Model**
   - **Model Used:** `meta-llama/Llama-3.1-8B` from Hugging Face.
   - **Tokenization and Generation:** Uses `AutoTokenizer` for tokenizing inputs and `AutoModelForCausalLM` for generating outputs.
   - **Environment:** The model loads onto a GPU if available; otherwise, it runs on the CPU.

### 4. **Key Functions**
   - **`create_baseline_prompt()`**: Constructs a prompt using novice and expert example pairs, along with the current novice caption. This prompt guides the model in producing an expert-level description.
   - **`generate_expert_description()`**: Generates the expert description based on the constructed prompt and specified parameters for generation.

### 5. **Important Parameters**
   - `max_new_tokens`: Sets the maximum number of new tokens the model generates for the expert description (default is 256).
   - `temperature`: Controls the randomness of the generation (default is 0.7). Lower values make the output more focused, while higher values increase creativity and diversity.
   - `batch_size`: Defines the number of examples processed per batch (default is 10).
   - `pad_token_id`: Avoids warnings by specifying the padding token ID to match the tokenizer's end-of-sequence (EOS) token ID.

### 6. **Execution and Output**
   - The code reads the novice descriptions in batches, applies the transformation, and saves the updated descriptions back into a CSV file named `musiccaps-baseline-expert-llama.csv`.



# metris.py

This code evaluates various metrics for text generation models, specifically focused on comparing novice-to-expert music caption transformations. Here’s a breakdown of the code's function, input, output, and important components:

### 1. **Functionality and Purpose**
   - This code calculates evaluation metrics for generated captions, including BLEU, METEOR, ROUGE, lexical diversity (TTR and MTLD), reading ease, vocabulary novelty, and caption novelty.
   - It reads data from a CSV file containing novice captions, ground truths, and generated expert captions. It then uses this data to compute metrics that assess the quality, diversity, and novelty of the generated expert-level captions.

### 2. **Input and Output**
   - **Input:**
      - **Data Files:**
         - `musiccaps-baseline-expert.csv`: A CSV file containing columns for novice descriptions (`novice`), generated expert descriptions (`gen_expert`), ground-truth expert descriptions (`caption`), and a flag (`is_balanced_subset`) indicating training vs. test subsets.
      - **Hand-curated Data Exclusion List:** `hand_curated` contains specific row indices to exclude from the data for more controlled metric calculations.
   - **Output:**
      - **Printed Results:** A dictionary `results` containing metric scores for various evaluation methods.

### 3. **Key Metrics and Calculations**
   - **BLEU (Bilingual Evaluation Understudy):** Measures the overlap of n-grams between predictions and ground truths. Calculated with different n-gram orders (`order=1,2,3,4`).
   - **METEOR:** Considers synonyms and stemming, making it sensitive to minor variations in wording.
   - **ROUGE-L:** Computes the longest common subsequence, emphasizing the recall of important words.
   - **Type-Token Ratio (TTR) and MTLD:** Measures lexical diversity in generated text.
   - **Flesch Reading Ease (FRE):** Calculates the readability of generated captions.
   - **Vocabulary Size and Novelty:** Assesses unique vocabulary usage in generated captions vs. training data.
   - **Caption Novelty:** Measures the proportion of new captions that don’t appear in the training set.

### 4. **Important Functions**
   - **`bleu()`, `meteor()`, `rouge()`:** Compute BLEU, METEOR, and ROUGE scores, leveraging the Hugging Face `evaluate` library.
   - **`vocab_novelty()`**: Computes vocabulary size and novelty by comparing generated vocab with training vocab.
   - **`caption_novelty()`**: Assesses how unique generated captions are compared to training captions.
   - **`ttr_and_MTLD()`**: Computes TTR and MTLD for lexical diversity.
   - **`fre()`**: Calculates the Flesch Reading Ease score for readability.

### 5. **Execution Flow**
   - The code reads the CSV file, prepares a training and test subset, computes each metric, and aggregates results into a dictionary (`results`), which is printed.
   
### 6. **Key Variables and Data Splits**
   - **`df_train` and `df_test`**: Subsets of the data used for training and testing.
   - **`predictions`, `ground_truths`, `predictions_test`, `tr_ground_truths`**: Lists used as inputs for the metric functions.




