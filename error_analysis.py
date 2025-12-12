import argparse
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='The dataset root folder', default='aae2/adus')
parser.add_argument('--model_name', type=str, help='LLM Model name or path', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
parser.add_argument('--epochs', type=int, help='number of epochs', default=1)
parser.add_argument('--batch_size', type=int, help='train batch size', default=8)
parser.add_argument('--gradient_steps', type=int, help='gradient_accumulation_steps', default=1)
parser.add_argument('--lora_r', type=int, help='lora rank', default=16)
parser.add_argument('--max_errors', type=int, help='max errors to display', default=20)

args = parser.parse_args()

# Build output directory path (same logic as training script)
use_lora = True
output_extra_detail = f"lora-r{args.lora_r}-{''.join(['q', 'k', 'v', 'g', 'u', 'd'])}"
output_extra_detail += f"-bs{args.batch_size}"
output_extra_detail += f"-ac{args.gradient_steps}"
output_extra_detail += f"-e{args.epochs}"

gold_path = f'datasets/{args.dataset}/test'
pred_dir = f'preds/{args.model_name.replace("/", "-")}-{output_extra_detail}-{args.dataset}'

if not os.path.exists(pred_dir):
    print(f"Error: Prediction directory not found: {pred_dir}")
    exit(1)

if not os.path.exists(gold_path):
    print(f"Error: Gold labels directory not found: {gold_path}")
    exit(1)

print(f"Gold path: {gold_path}")
print(f"Prediction path: {pred_dir}")
print()

def decompose(text):
    """Extract tags from text using regex pattern"""
    pattern = r'<(\w+)>([^<]+)<\/\w+>'
    matches = re.findall(pattern, text)
    return matches

def calculate_tags(text, tag_matches):
    """Map extracted tags to token positions"""
    if not tag_matches:
        return [(i, 'O') for i in range(len(text.split()))]
    
    tokens = text.split()
    token_labels = ['O'] * len(tokens)
    
    for tag_type, tag_text in tag_matches:
        try:
            # Use re.escape to handle special characters in tag_text
            pattern = re.escape(tag_text)
            match = re.search(pattern, text)
            
            if not match:
                continue
            
            # Find which tokens this tag spans
            char_pos = match.start()
            token_pos = 0
            char_count = 0
            
            for i, token in enumerate(tokens):
                if char_count + len(token) > char_pos:
                    token_pos = i
                    break
                char_count += len(token) + 1  # +1 for space
            
            # Mark tokens with this tag
            tag_word_count = len(tag_text.split())
            for j in range(token_pos, min(token_pos + tag_word_count, len(tokens))):
                if j < len(tokens):
                    token_labels[j] = tag_type
        except Exception as e:
            continue
    
    return list(enumerate(token_labels))

def extract_tags(gold_sample, pred_sample):
    """Extract and align tags from gold and prediction"""
    try:
        gold_text, gold_content = gold_sample
        pred_text, pred_content = pred_sample
        
        gold_matches = decompose(gold_content)
        pred_matches = decompose(pred_content)
        
        # Get the actual text (without tags) for tokenization
        gold_text_clean = re.sub(r'<[^>]+>|</[^>]+>', '', gold_content).strip()
        
        gold_labels = calculate_tags(gold_text_clean, gold_matches)
        pred_labels = calculate_tags(gold_text_clean, pred_matches)
        
        gold_tags = [label for _, label in gold_labels]
        pred_tags = [label for _, label in pred_labels]
        
        return (gold_tags, pred_tags, gold_text_clean)
    except Exception as e:
        return None

# Load samples
gold_samples = []
pred_samples = []

for gold_file in sorted(os.listdir(gold_path)):
    if not gold_file.endswith('.json'):
        continue
    
    with open(f'{gold_path}/{gold_file}') as f:
        gold_samples.append((gold_file, f.read()))

for pred_file in sorted(os.listdir(pred_dir)):
    if not pred_file.endswith('.txt'):
        continue
    
    with open(f'{pred_dir}/{pred_file}') as f:
        pred_samples.append((pred_file.replace('.txt', '.json'), f.read()))

print(f"Gold samples: {len(gold_samples)}")
print(f"Prediction samples: {len(pred_samples)}")
print()

# Match and analyze errors
error_stats = defaultdict(int)
errors = []

for gold_sample, pred_sample in tqdm(zip(gold_samples, pred_samples)):
    result = extract_tags(gold_sample, pred_sample)
    
    if result is None:
        continue
    
    gold_tags, pred_tags, text = result
    tokens = text.split()
    
    # Check for mismatches
    for i, (gold_tag, pred_tag) in enumerate(zip(gold_tags, pred_tags)):
        if gold_tag != pred_tag:
            error_type = ""
            if gold_tag == 'O' and pred_tag != 'O':
                error_type = "False Positive"
                error_stats["False Positive"] += 1
            elif gold_tag != 'O' and pred_tag == 'O':
                error_type = "False Negative"
                error_stats["False Negative"] += 1
            else:
                error_type = "Wrong Label"
                error_stats["Wrong Label"] += 1
            
            # Store error with context
            context_start = max(0, i - 2)
            context_end = min(len(tokens), i + 3)
            context_tokens = tokens[context_start:context_end]
            
            errors.append({
                'file': gold_sample[0],
                'token_idx': i,
                'token': tokens[i],
                'gold': gold_tag,
                'pred': pred_tag,
                'type': error_type,
                'context': ' '.join(context_tokens),
                'context_start_idx': context_start
            })

# Print statistics
print("\n" + "="*80)
print("ERROR ANALYSIS SUMMARY")
print("="*80)
print(f"\nError Type Distribution:")
for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {error_type}: {count}")

total_errors = sum(error_stats.values())
print(f"\nTotal Errors: {total_errors}")

# Sort errors by type for better visualization
errors_by_type = defaultdict(list)
for error in errors:
    errors_by_type[error['type']].append(error)

# Display sample errors
for error_type in sorted(error_stats.keys()):
    print(f"\n{'='*80}")
    print(f"Sample {error_type} Errors (showing up to {args.max_errors}):")
    print("="*80)
    
    displayed = 0
    for error in errors_by_type[error_type]:
        if displayed >= args.max_errors:
            remaining = len(errors_by_type[error_type]) - displayed
            if remaining > 0:
                print(f"\n... and {remaining} more {error_type} errors")
            break
        
        print(f"\nFile: {error['file']}")
        print(f"Token: '{error['token']}'")
        print(f"Gold: {error['gold']} | Prediction: {error['pred']}")
        print(f"Context: ...{error['context']}...")
        print(f"Position in context: token #{error['token_idx'] - error['context_start_idx']}")
        
        displayed += 1

print("\n" + "="*80)
