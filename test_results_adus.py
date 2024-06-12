import os
import re
from typing import List, Tuple

from joblib import Parallel, delayed
from rouge.rouge_score import rouge_l_summary_level
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm


from settings import Settings

dataset = 'aae2/relations'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
use_lora = True

settings = Settings(
    dataset_path = f'data/processed/{dataset}',
    per_device_train_batch_size = 1,
    # model_name = 'models/microsoft/phi-2',
    model_name = model_name,
    # output_dir = 'output/phi-28bitqlora',
    output_dir = f'output/{model_name.replace("/", "-")}{"-lora" if use_lora else ""}-{dataset}',
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
pred_path = f"preds/{settings.output_dir.replace('output/', '')}"


def decompose(name, output: str) -> List[Tuple[str, str]]:
    result = []
    claims = re.findall('(?<=<claim>)(.*?)(?=<\/claim>)', output,  re.MULTILINE | re.IGNORECASE)
    major_claims = re.findall('(?<=<majorclaim>)(.*?)(?=<\/majorclaim>)', output,  re.MULTILINE | re.IGNORECASE)
    premises = re.findall('(?<=<premise>)(.*?)(?=<\/premise>)', output,  re.MULTILINE | re.IGNORECASE)
    for c in claims:
        result.append(('Claim', c))
    for mc in major_claims:
        result.append(('MajorClaim', mc))
    for p in premises:
        result.append(('Premise', p))
    print(name, len(result))
    print(major_claims)
    print(claims)
    print(premises)
    return result


def calculate_scores(golds, preds):
    golds_labels = []
    preds_labels = ['O'] * len(golds)
    matched_preds_text = []
    matched_preds_text_label = []
    for j, g in enumerate(golds):
        golds_labels.append(g[0])
        for i, p in enumerate(preds):
            lcs = rouge_l_summary_level(g[1], p[1])
            if lcs['f'] > 0.9 and i not in matched_preds_text: #check if already not matched
                matched_preds_text.append(i)
                if p[0] == g[0]:
                    preds_labels[j] = g[0]
                    matched_preds_text_label.append(i)
    text_acc = len(matched_preds_text)/len(golds)
    text_label_acc = len(matched_preds_text_label)/len(golds)
    return text_acc, text_label_acc, golds_labels, preds_labels


text_accs = []
text_label_accs = []


def calculate_metrics(gold, pred):
    pred_adus = decompose(pred[0], pred[1])
    gold_adus = decompose(gold[0], gold[1])
    if len(gold_adus) == 0: # not all paragraphs have adus
        return
    text_acc, text_label_acc, golds_labels, preds_labels = calculate_scores(gold_adus, pred_adus)
    return text_acc, text_label_acc, golds_labels, preds_labels


gold_samples = []
pred_samples = []

for (gold, pred) in tqdm(zip(sorted(os.listdir(gold_path)), sorted(os.listdir(pred_path))), total = len(os.listdir(pred_path))):
    with open(f'{gold_path}/{gold}') as g, open(f'{pred_path}/{pred}') as p:
        gold_samples.append((gold, g.read()))
        pred_samples.append((pred, p.read()))

def flatten(xss):
    return [x for xs in xss for x in xs]



results = Parallel(n_jobs=1)(delayed(calculate_metrics)(g,p) for g,p in zip(gold_samples, pred_samples))
text_accs = [r[0] for r in results if r is not None]
text_label_accs = [r[1]for r in results if r is not None]
golds_labels = flatten([r[2] for r in results if r is not None])
preds_labels = flatten([r[3] for r in results if r is not None])


score = f1_score(golds_labels, preds_labels, average='macro',)
conf_mat = confusion_matrix(golds_labels, preds_labels, labels=["MajorClaim", "Claim", "Premise", "O"])
print(score)
print(conf_mat)
preds_labels = [l if l != 'O' else 'Premise' for l in preds_labels]
score = f1_score(golds_labels, preds_labels, average='macro',)
conf_mat = confusion_matrix(golds_labels, preds_labels, labels=["MajorClaim", "Claim", "Premise"])
print(score)
print(conf_mat)