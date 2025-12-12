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
parser.add_argument('--bit4', action='store_true', help='user 4bit quantization', default=False)
parser.add_argument('--bit8', action='store_true', help='user 8bit quantization', default=False)
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
output_extra_detail += "-fp" if (not (args.bit4 and args.bit8)) else ""


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
    """Extract ADUs (tags) from text using regex pattern"""
    pattern = r'<(\w+)>([^<]+)<\/\w+>'
    matches = re.findall(pattern, text)
    return matches

def get_adu_segments(text):
    """Extract ADU segments with their labels and text"""
    pattern = r'<(\w+)>([^<]+)<\/\w+>'
    matches = re.findall(pattern, text)
    
    segments = []
    for label, content in matches:
        # Normalize the content (strip whitespace)
        content_normalized = content.strip()
        segments.append({
            'label': label,
            'text': content_normalized,
            'text_lower': content_normalized.lower()
        })
    
    return segments

def compare_segments(gold_segments, pred_segments):
    """Compare gold and prediction segments, identify errors"""
    errors = []
    matched_pred = set()
    
    # Check each gold segment
    for gold_seg in gold_segments:
        # Try to find exact match in predictions
        found = False
        for i, pred_seg in enumerate(pred_segments):
            if i not in matched_pred and gold_seg['label'] == pred_seg['label'] and \
               gold_seg['text_lower'] == pred_seg['text_lower']:
                found = True
                matched_pred.add(i)
                break
        
        if not found:
            # Check if there's a partial match (same text, wrong label or vice versa)
            partial_match = None
            for i, pred_seg in enumerate(pred_segments):
                if i not in matched_pred and gold_seg['text_lower'] == pred_seg['text_lower']:
                    partial_match = pred_seg
                    matched_pred.add(i)
                    break
            
            if partial_match:
                errors.append({
                    'type': 'Wrong Label',
                    'gold_label': gold_seg['label'],
                    'pred_label': partial_match['label'],
                    'text': gold_seg['text']
                })
            else:
                errors.append({
                    'type': 'False Negative',
                    'gold_label': gold_seg['label'],
                    'pred_label': None,
                    'text': gold_seg['text']
                })
    
    # Check for false positives (predictions with no gold match)
    for i, pred_seg in enumerate(pred_segments):
        if i not in matched_pred:
            errors.append({
                'type': 'False Positive',
                'gold_label': None,
                'pred_label': pred_seg['label'],
                'text': pred_seg['text']
            })
    
    return errors

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
errors_list = []

for gold_sample, pred_sample in tqdm(zip(gold_samples, pred_samples)):
    try:
        gold_file, gold_content = gold_sample
        pred_file, pred_content = pred_sample
        
        # Extract segments from both
        gold_segments = get_adu_segments(gold_content)
        pred_segments = get_adu_segments(pred_content)
        
        # Compare segments
        segment_errors = compare_segments(gold_segments, pred_segments)
        
        # If there are any errors in this sample, store it
        if segment_errors:
            errors_list.append({
                'file': gold_file,
                'errors': segment_errors,
                'gold_text': gold_content,
                'pred_text': pred_content
            })
            
            for error in segment_errors:
                error_stats[error['type']] += 1
    except Exception as e:
        continue

# Print statistics
print("\n" + "="*80)
print("ERROR ANALYSIS SUMMARY - ADU Level")
print("="*80)
print(f"\nDocuments with errors: {len(errors_list)}")
print(f"\nError Type Distribution:")
for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {error_type}: {count}")

total_errors = sum(error_stats.values())
print(f"\nTotal ADU Errors: {total_errors}")

# Sort errors by type
errors_by_type = defaultdict(list)
for error_sample in errors_list:
    for error in error_sample['errors']:
        errors_by_type[error['type']].append({
            'file': error_sample['file'],
            **error
        })

# Display sample errors
for error_type in sorted(error_stats.keys()):
    print(f"\n{'='*80}")
    print(f"Sample {error_type} Cases (showing up to {args.max_errors}):")
    print("="*80)
    
    displayed = 0
    for error in errors_by_type[error_type]:
        if displayed >= args.max_errors:
            remaining = len(errors_by_type[error_type]) - displayed
            if remaining > 0:
                print(f"\n... and {remaining} more {error_type} cases")
            break
        
        print(f"\nFile: {error['file']}")
        print(f"Text: \"{error['text']}\"")
        
        if error_type == 'False Negative':
            print(f"Missing in predictions - Should be labeled as: {error['gold_label']}")
        elif error_type == 'False Positive':
            print(f"Incorrectly predicted - Labeled as: {error['pred_label']} (should not be present)")
        elif error_type == 'Wrong Label':
            print(f"Gold label: {error['gold_label']} | Predicted label: {error['pred_label']}")
        
        displayed += 1

print("\n" + "="*80)
