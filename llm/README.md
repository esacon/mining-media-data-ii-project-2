# LLM Critical Error Detection

Zero-shot and few-shot critical error detection using small language models as an alternative to fine-tuned DistilBERT.

## Models

- **Llama3** (Llama-3.1-8B-Instruct, 70B params)
- **Phi3** (Phi-3-mini-4k-instruct, 67M params) 
- **Mixtral** (Mixtral-8x7B-Instruct-v0.1, 128B params)

## Prompts

### Zero-shot
Direct WMT21 error category instructions without examples.

### Few-shot  
7 examples covering WMT21 categories: toxicity, safety, named entities, sentiment, numbers.

## WMT21 Critical Errors

1. **Toxicity**: Hate speech/profanity changes
2. **Safety**: Health/safety information changes  
3. **Named Entities**: Wrong names/places/organizations
4. **Sentiment**: Negation/sentiment changes
5. **Numbers**: Wrong numbers/dates/units

## Usage

```bash
# Single model evaluation
make evaluate-llm MODEL=phi3 PROMPT=zero_shot LANG=en-de

# All models and prompts 
make evaluate-llm MODEL=all PROMPT=all LANG=en-de

# Quick test
make debug-llm LANG=en-de
```

## Training All Languages

```bash
# Train on all language pairs with all models
for lang in en-de en-ja en-zh en-cs; do
  make evaluate-llm MODEL=all PROMPT=all LANG=$lang
done
```

## Output

Results saved to `results/llm_evaluation/`:
- Individual: `llm_results_{model}_{prompt}_{lang}.json`
- Summary: `llm_evaluation_summary_{lang}.json`

Each includes accuracy, MCC, precision, recall, F1 scores plus sample predictions.
