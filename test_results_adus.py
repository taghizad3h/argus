import json
import os
import re
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

# dataset = 'aae2/adus'
dataset = 'argmicro/adus'
# dataset = 'pe2-adus-embedded-paragraph-level'
# model_name = 'NousResearch/Llama-2-7b-chat-hf'
# model_name = 'unsloth/llama-2-7b'Saleh but assistant
# model_name = 'unsloth/mistral-7b'
# model_name = 'unsloth/llama-3-8b-Instruct'
# model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# model_name = 'unsloth/Llama-3.2-1B-Instruct'
# model_name = 'unsloth/Llama-3.2-3B-Instruct'
model_name = 'unsloth/Meta-Llama-3.1-8B-Instruct'
config_name = 'lora-r16-qkvgud-bs8-ac1-e20-fp' #r = rank of lora g=gate-proj u=up-proj d=down-proj fp = full presicion

settings = Settings(
    dataset_path = f'datasets/{dataset}',
    per_device_train_batch_size = 1,
    # model_name = 'models/microsoft/phi-2',
    model_name = model_name,
    # output_dir = 'output/phi-28bitqlora',
    output_dir = f'output/{model_name.replace("/", "-")}-{config_name}-{dataset}',
    use_4bit = False,
    use_8bit = False,
    gradient_accumulation_steps = 4,
    llm_int8_enable_fp32_cpu_offload = True,
    per_device_eval_batch_size = 4,
    num_train_epochs=10,
    max_seq_length=1024,
    save_steps = 100
)

gold_path = f'{settings.dataset_path}/test'
pred_path = settings.output_dir.replace('output/', 'preds/')


def convert_to_bio(text, labels: list = ['MajorClaim', 'Claim', 'Premise']):
    clean_text = re.sub(r'<(/?\w+)>', ' ', text, flags = re.MULTILINE|re.IGNORECASE)
    clean_text = ' '.join(clean_text.split())    
    tags_pattern = re.compile(r'<(\w+)>([\w\s]+)<\/\w+>')
    tags = ['O'] * len(clean_text.split())
    last_start_index = -1
    for match in tags_pattern.finditer(text):
        tag = match.group(1)
        tag_text = match.group(2).rstrip()
        char_start_indexs = [m.start() for m in re.finditer(tag_text, clean_text)]
        for csi in char_start_indexs: #some times same argumet exists in multiple spans
            start_index = len(clean_text[:csi].split())
            if start_index > last_start_index:
                last_start_index = start_index
                break
        end_index = start_index + len(tag_text.split())
        for i in range(start_index, end_index):
            tags[i] = tag
            
    return tags


def decompose(name, output: str) -> List[Tuple[str, str]]:
    result = []
    if 'This sentence is not argumentative' in output:
        return result
    
    tags_pattern = re.compile(r'<(\w+)>([\w\s]+)<\/\w+>')
    for match in tags_pattern.finditer(output):
        tag = match.group(1)
        tag_text = match.group(2).rstrip()
        if tag_text is not None and tag_text.strip() != '':
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



def calculate_tags(golds, preds, gold_annotated_sentence, target_sentence):
    gold_tags = ['O'] * len(target_sentence.split())
    pred_tags = ['O'] * len(target_sentence.split())
    # print(f'len of tags {len(pred_tags)}')
    
    for g in golds:
        gtext = g[1]
        glabel = g[0]
        # print(f'the gold text is {gtext.strip()}')
        char_start_index = [m.start() for m in re.finditer(gtext.strip(), target_sentence, flags=re.IGNORECASE|re.MULTILINE)][0]
        start_index = len(target_sentence[:char_start_index].split())
        end_index = start_index +  len(gtext.split())
        for i in range(start_index, end_index):
            gold_tags[i] = glabel
    
    for p in preds:
        ptext = p[1]
        # print(f'the text is {ptext}')
        plabel = p[0]
        # print(f'the label is {plabel}')
        char_start_indexes = [m.start() for m in re.finditer(ptext.strip(), target_sentence, flags=re.IGNORECASE|re.MULTILINE)]
        if len(char_start_indexes) != 1:
            continue
        # print(f'len of chat start index {len(char_start_indexes)}')
        char_start_index = char_start_indexes[0]
        start_index = len(target_sentence[:char_start_index].split())
        end_index = start_index +  len(ptext.strip().split())
        for i in range(start_index, end_index):
            # print(i)
            pred_tags[i] = plabel
            
    return gold_tags, pred_tags


text_accs = []
text_label_accs = []


def extract_tags(gold, pred):
    pred_adus = decompose(pred[0], pred[1])
    pred_adus = refine_preds(pred_adus)
    gold_adus = decompose(gold[0], gold[1])
    gold_adus_dict = json.loads(gold[1])
    gold_annotated_sentence = gold_adus_dict[2]['content'].replace('The annotated format of given sentence is:\n', '')
    # gold_annotated_sentence = gold_adus_dict[2]['content'].replace('The annotated format of given paragraph is:\n', '')
    target_sentence = re.findall(r'(?<=What argument components exists in sentence \")([.\s\S]*)(?=\" from the above dispute)', gold_adus_dict[1]['content'])[0]
    # target_sentence = ' '.join(gold_adus_dict[1]['content'].split('\n')[2:])
    golds_labels, preds_labels = calculate_tags(gold_adus, pred_adus, gold_annotated_sentence, target_sentence)
    return golds_labels, preds_labels


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


def convert_to_bio(labels):
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