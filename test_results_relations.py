import json
import os

from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm

from settings import Settings

# dataset = 'aae2/relations'
dataset = 'argmicro/relations'

# model_name = 'unsloth/llama-3-8b-Instruct'
model_name = 'unsloth/Meta-Llama-3.1-8B-Instruct'

config_name = 'lora-r16-q-bs8-ac1-e10-fp' #r = rank of lora g=gate-proj u=up-proj d=down-proj fp = full presicion

use_lora = True

settings = Settings(
    dataset_path = f'datasets/{dataset}',
    per_device_train_batch_size = 1,
    model_name = model_name,
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

# relations = [
#     "1. attacks",
#     "2. supports",
#     "3. no relation"
# ]

relations = [
    "1. reb",
    "2. und",
    "3. joint",
    "4. exa",
    "5. sup",
    "6. no relation"
]


response_template = 'assistant<|end_header_id|>'

if 'tiny' in settings.model_name.lower():
    response_template = "<|assistant|>"
elif 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower():
    response_template = "[/INST]"
elif 'phi' in settings.model_name.lower():
    response_template = "Output:"
elif 'llama-3' in settings.model_name.lower():
    response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'


gold_path = f'{settings.dataset_path}/test'
pred_path = f"preds/{settings.output_dir.replace('output/', '')}"


gold_labels = []
pred_labels = []



for (gold, pred) in tqdm(zip(sorted(os.listdir(gold_path)), sorted(os.listdir(pred_path))), total = len(os.listdir(pred_path))):
    with open(f'{gold_path}/{gold}') as g, open(f'{pred_path}/{pred}') as p:
        gold_label = ''
        gold = json.load(g)
        for item in gold:
            if item['role'] == 'assistant':
                gold_label = item['content']
        if gold_label == '':
            raise Exception('gold label does not exists')
        pred_text = p.read()
        pred_label = pred_text.split(response_template)[1].strip()
        
#         if pred_label.startswith('Major'):
#             pred_label = "MajorClaim"
#         elif pred_label.startswith('Claim'):
#             pred_label = "Claim"
#         else:
#             pred_label = "Premise"
        found = False
        for r in relations:
            if pred_label.startswith(r[0]):
                found = True
                pred_label = r
                break
        if not found:
            pred_label = relations[-1]
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)


score = f1_score(gold_labels, pred_labels, average='macro')
# conf_mat = confusion_matrix(gold_labels, pred_labels, labels=["MajorClaim", "Claim", "Premise"])
conf_mat = confusion_matrix(gold_labels, pred_labels, labels=relations)

print(score)
print(conf_mat)
print(classification_report(gold_labels, pred_labels))