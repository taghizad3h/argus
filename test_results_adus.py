import json
import os
import re
import argparse
from typing import List, Tuple

from joblib import Parallel, delayed
from rouge.rouge_score import rouge_l_summary_level
from seqeval.metrics import accuracy_score as saccuracy_score
from seqeval.metrics import classification_report as sclassification_report
from seqeval.metrics import f1_score as sf1_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from seqeval.scheme import IOB2, IOB1

from settings import Settings

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='The dataset root folder', default='aae2/adus')
parser.add_argument('--model_name', type=str, help='LLM Model name or path', default='unsloth/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--epochs', type=int, help='number of epochs', default=1)
parser.add_argument('--batch_size', type=int, help='train batch size', default=8)
parser.add_argument('--gradient_steps', type=int, help='gradient_accumulation_steps', default=1)
parser.add_argument('--lora_r', type=int, help='lora rank', default=16)
parser.add_argument('--lora_modules', type=str, help='lora modules', default='q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj')
parser.add_argument('--bit4', action='store_true', help='user 4bit quantization', default=False)
parser.add_argument('--bit8', action='store_true', help='user 8bit quantization', default=False)

args = parser.parse_args()

# Build config name
output_extra_detail = ''
output_extra_detail += f"lora-r{args.lora_r}-{''.join([r[0] for r in args.lora_modules.split(',')])}"
output_extra_detail += f"-bs{args.batch_size}"
output_extra_detail += f"-ac{args.gradient_steps}"
output_extra_detail += f"-e{args.epochs}"
output_extra_detail += "-q4" if args.bit4 else ""
output_extra_detail += "-q8" if args.bit8 else ""
output_extra_detail += "-fp" if (not (args.bit4 and args.bit8)) else ""

dataset = args.dataset
model_name = args.model_name
config_name = output_extra_detail

settings = Settings(
    dataset_path=f'datasets/{dataset}',
    per_device_train_batch_size=args.batch_size,
    model_name=model_name,
    output_dir=f'output/{model_name.replace("/", "-")}-{config_name}-{dataset}',
    use_4bit=args.bit4,
    use_8bit=args.bit8,
    gradient_accumulation_steps=args.gradient_steps,
    llm_int8_enable_fp32_cpu_offload=True,
    per_device_eval_batch_size=4,
    num_train_epochs=args.epochs,
    max_seq_length=1024,
    save_steps=1000,
    load_in_4bit=args.bit4,
    lora_r=args.lora_r
)

gold_path = f'{settings.dataset_path}/test'
pred_path = settings.output_dir.replace('output/', 'preds/')


def decompose(name, output: str) -> List[Tuple[str, str]]:
    """Extract tagged components from the model output.
    Note: output now contains only the assistant message (no full conversation).
    """
    result = []
    
    # Check for non-argumentative response
    if 'not argumentative' in output.lower() or 'This sentence is not argumentative' in output:
        return result
    
    # Extract all tagged components (improved regex to handle any content)
    tags_pattern = re.compile(r'<(\w+)>([^<]+)<\/\w+>')
    for match in tags_pattern.finditer(output):
        tag = match.group(1)
        tag_text = match.group(2).rstrip()
        if tag_text and tag_text.strip():
            result.append((tag, tag_text))
    
    return result


def refine_preds(preds):
    refined = []
    
    def already_exists(text):
        for r in refined:
            lcs = rouge_l_summary_level(r[1], text)
            if lcs['f'] > 0.9: #check if already not matched
                return True
        return False
    
    for p1 in preds:
        if not already_exists(p1[1]):
            refined.append(p1)
    return refined


def convert_to_bio(labels):
    """Convert sequence labels to BIO format: B- for first occurrence, I- for subsequent."""
    bio_labels = []
    prev_label = 'O'
    
    for label in labels:
        if label == 'O':
            bio_labels.append('O')
        elif label == prev_label:
            bio_labels.append('I-' + label)
        else:
            bio_labels.append('B-' + label)
        
        prev_label = label

    return bio_labels



def calculate_tags(golds, preds, gold_annotated_sentence, target_sentence):
    gold_tags = ['O'] * len(target_sentence.split())
    pred_tags = ['O'] * len(target_sentence.split())
    
    # Process gold labels
    for g in golds:
        gtext = g[1].strip()
        glabel = g[0]
        try:
            char_start_indexes = [m.start() for m in re.finditer(re.escape(gtext), target_sentence, flags=re.IGNORECASE|re.MULTILINE)]
            if not char_start_indexes:
                # Try partial matching if exact match fails
                continue
            char_start_index = char_start_indexes[0]
            start_index = len(target_sentence[:char_start_index].split())
            end_index = start_index + len(gtext.split())
            # Ensure we don't go out of bounds
            end_index = min(end_index, len(gold_tags))
            for i in range(start_index, end_index):
                gold_tags[i] = glabel
        except Exception as e:
            # Skip this annotation if there's an error finding it
            continue
    
    # Process predicted labels
    for p in preds:
        ptext = p[1].strip()
        plabel = p[0]
        try:
            char_start_indexes = [m.start() for m in re.finditer(re.escape(ptext), target_sentence, flags=re.IGNORECASE|re.MULTILINE)]
            if len(char_start_indexes) != 1:
                # Skip if we find 0 or multiple matches
                continue
            
            char_start_index = char_start_indexes[0]
            start_index = len(target_sentence[:char_start_index].split())
            end_index = start_index + len(ptext.split())
            # Ensure we don't go out of bounds
            end_index = min(end_index, len(pred_tags))
            for i in range(start_index, end_index):
                pred_tags[i] = plabel
        except Exception as e:
            # Skip this annotation if there's an error
            continue
            
    return gold_tags, pred_tags


text_accs = []
text_label_accs = []


def extract_tags(gold, pred):
    """Extract and align gold and predicted tags.
    Returns None if there's an error processing the sample.
    """
    try:
        pred_adus = decompose(pred[0], pred[1])
        pred_adus = refine_preds(pred_adus)
        gold_adus = decompose(gold[0], gold[1])
        gold_adus_dict = json.loads(gold[1])
        gold_annotated_sentence = gold_adus_dict[2]['content'].replace('The annotated format of given sentence is:\n', '')
        # gold_annotated_sentence = gold_adus_dict[2]['content'].replace('The annotated format of given paragraph is:\n', '')
        target_sentence_matches = re.findall(r'(?<=What argument components exists in sentence \")([.\s\S]*)(?=\" from the above dispute)', gold_adus_dict[1]['content'])
        
        if not target_sentence_matches:
            return None
        
        target_sentence = target_sentence_matches[0]
        golds_labels, preds_labels = calculate_tags(gold_adus, pred_adus, gold_annotated_sentence, target_sentence)
        return golds_labels, preds_labels
    except Exception as e:
        print(f"Error processing {gold[0]}: {str(e)}")
        return None


gold_samples = []
pred_samples = []

for (gold, pred) in tqdm(zip(sorted(os.listdir(gold_path)), sorted(os.listdir(pred_path))), total = len(os.listdir(pred_path))):
    with open(f'{gold_path}/{gold}') as g, open(f'{pred_path}/{pred}') as p:
        gold_samples.append((gold, g.read()))
        pred_samples.append((pred, p.read()))

def flatten(xss):
    return [x for xs in xss for x in xs]


gold_samples = sorted(gold_samples, key=lambda x : x[0])
pred_samples = sorted(pred_samples, key=lambda x : x[0])

print(len(gold_samples))
print(len(pred_samples))

# for g, p in zip(gold_samples, pred_samples):
#     print(g[0])
#     print(p[0])
#     extract_tags(g, p)
    

results = Parallel(n_jobs=1)(delayed(extract_tags)(g,p) for g,p in zip(gold_samples, pred_samples))

y_true = [r[0] for r in results if r is not None]
y_pred = [r[1] for r in results if r is not None]

bio_true = [convert_to_bio(y) for y in y_true]
bio_pred = [convert_to_bio(y) for y in y_pred]
            

print(sf1_score(bio_true, bio_pred))
print(sclassification_report(bio_true, bio_pred))

print(sf1_score(bio_true, bio_pred, mode='strict', scheme=IOB2))
print(sclassification_report(bio_true, bio_pred, mode='strict', scheme=IOB2))


golds_labels = flatten([r[0] for r in results if r is not None])
preds_labels = flatten([r[1] for r in results if r is not None])


score = f1_score(golds_labels, preds_labels, average='macro',)
conf_mat = confusion_matrix(golds_labels, preds_labels, labels=["MajorClaim", "Claim", "Premise", "O"])
print(score)
print(conf_mat)
print(classification_report(golds_labels, preds_labels))


score = f1_score(golds_labels, preds_labels, average='macro', labels=["MajorClaim", "Claim", "Premise"])
conf_mat = confusion_matrix(golds_labels, preds_labels, labels=["MajorClaim", "Claim", "Premise"])
print(score)
print(conf_mat)
print(classification_report(golds_labels, preds_labels, labels=["MajorClaim", "Claim", "Premise"]))