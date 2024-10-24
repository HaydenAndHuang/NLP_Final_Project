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

# pair_gen.ipynb
和锴锴发到群上的 code 一样，set up the visual env first and run the code. 

# pair_gen.py (don't run, not ready)
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

