#%%
import random
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MBartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import evaluate
import numpy as np

#%%
# Setup seeds
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# for using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.DataFrame(columns = ['원문','번역문'])
path = './한영번역/'

file_list = [ '2_대화체.xlsx',
 '1_구어체(2).xlsx',
 '1_구어체(1).xlsx',
 '3_문어체_뉴스(2).xlsx',
 '3_문어체_뉴스(3).xlsx',
 '3_문어체_뉴스(1).xlsx',
 '4_문어체_한국문화.xlsx',
 '5_문어체_조례.xlsx',
 '3_문어체_뉴스(4).xlsx',
 '6_문어체_지자체웹사이트.xlsx']


for file in file_list:
    temp = pd.read_excel(path+file)
    df = pd.concat([df,temp[['원문','번역문']]])

df.head()

df.rename(columns={"번역문": "SRC"}, errors="raise", inplace=True)
df.rename(columns={"원문": "TRG"}, errors="raise", inplace=True)


#%%
raw_eng = []
raw_kor = []

for row in df.iterrows():
    src_sentence = row[1]['SRC']
    trg_sentence = row[1]['TRG']

    src_sentence = src_sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    src_sentence = re.sub(r"([?.!,])", r" \1 ", src_sentence)
    src_sentence = re.sub(r'[" "]+', " ", src_sentence)
    # removing contractions
    src_sentence = re.sub(r"i'm", "i am", src_sentence)
    src_sentence = re.sub(r"he's", "he is", src_sentence)
    src_sentence = re.sub(r"she's", "she is", src_sentence)
    src_sentence = re.sub(r"it's", "it is", src_sentence)
    src_sentence = re.sub(r"that's", "that is", src_sentence)
    src_sentence = re.sub(r"what's", "that is", src_sentence)
    src_sentence = re.sub(r"where's", "where is", src_sentence)
    src_sentence = re.sub(r"how's", "how is", src_sentence)
    src_sentence = re.sub(r"\'ll", " will", src_sentence)
    src_sentence = re.sub(r"\'ve", " have", src_sentence)
    src_sentence = re.sub(r"\'re", " are", src_sentence)
    src_sentence = re.sub(r"\'d", " would", src_sentence)
    src_sentence = re.sub(r"\'re", " are", src_sentence)
    src_sentence = re.sub(r"won't", "will not", src_sentence)
    src_sentence = re.sub(r"can't", "cannot", src_sentence)
    src_sentence = re.sub(r"n't", " not", src_sentence)
    src_sentence = re.sub(r"n'", "ng", src_sentence)
    src_sentence = re.sub(r"'bout", "about", src_sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    src_sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", src_sentence)
    src_sentence = src_sentence.strip()

    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    trg_sentence = re.sub(r"([?.!,])", r" \1 ", trg_sentence)
    trg_sentence = trg_sentence.strip()

    if len(trg_sentence) >= 128:
        continue
    raw_eng.append(src_sentence)
    raw_kor.append(trg_sentence)

print(raw_eng[:5])
print(raw_kor[:5])

df1 = pd.DataFrame(raw_eng)
df2 = pd.DataFrame(raw_kor)

df1.rename(columns={0: "SRC"}, errors="raise", inplace=True)
df2.rename(columns={0: "TRG"}, errors="raise", inplace=True)

train_df = pd.concat([df1, df2], axis=1)

print('Translation Pair :',len(train_df)) # 리뷰 개수 출력

raw_src = train_df['SRC'].tolist()
raw_trg = train_df['TRG'].tolist()

train_df.to_csv(path+'/Translation_dataset_split.csv',index = False)

# %%
# load
path = './한영번역/'
train_df = pd.read_csv(path+'/Translation_dataset_split.csv')

# train_df = train_df.sample(n=1024*16, # number of items from axis to return.
#           random_state=1234) # seed for random number generator for reproducibility

# train_df = train_df.sample(n=1024*128*5, # number of items from axis to return.
#           random_state=1234) # seed for random number generator for reproducibility

dataset = Dataset.from_pandas(train_df)

# 80% train, 10% valid, 10% test
# train_testvalid = dataset.train_test_split(test_size=0.2)
# train_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
# train_test_valid_dataset = DatasetDict({
#     'train': train_testvalid['train'],
#     'test': train_valid['train'],
#     'valid': train_valid['test']})

train_test_valid_dataset = dataset.train_test_split(test_size=3000, seed=42)
train_test_valid_dataset = DatasetDict({
    'train': train_test_valid_dataset['train'],
    'valid': train_test_valid_dataset['test']})


# save
# train_test_valid_dataset.save_to_disk("AIHub_data_100000")

# load
# train_test_valid_dataset = DatasetDict.load_from_disk("AIHub_data_100000")


# %%

MODEL = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_sample_data(data):
  input_feature = tokenizer(data["SRC"], truncation=True, max_length=128)
  label = tokenizer(data["TRG"], truncation=True, max_length=128)
  return {
    "input_ids": input_feature["input_ids"],
    "attention_mask": input_feature["attention_mask"],
    "labels": label["input_ids"],
  }

tokenized = train_test_valid_dataset.map(
  tokenize_sample_data,
  remove_columns=["SRC", "TRG"],
  batched=True,
  batch_size=128)


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mbart_config = AutoConfig.from_pretrained(
  MODEL,
  max_length=128,
  length_penalty=0.6,
  no_repeat_ngram_size=2,
  num_beams=15,
)
model = (AutoModelForSeq2SeqLM
         .from_pretrained(MODEL, config=mbart_config)
         .to(device))

#%%

data_collator = DataCollatorForSeq2Seq(
  tokenizer,
  model=model,
  return_tensors="pt")

#%%
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
# %%
# Top 5 bleu score
sample_dataloader = DataLoader(
  tokenized["valid"].with_format("torch"),
  collate_fn=data_collator,
  batch_size=5)
for batch in sample_dataloader:
  with torch.no_grad():
    preds = model.generate(
      batch["input_ids"].to(device)
    )
  labels = batch["labels"]
  break

compute_metrics([preds, labels])
# %%
training_args = Seq2SeqTrainingArguments(
  output_dir = "result/nllb1.3b_test",
  log_level = "error",
  num_train_epochs = 1,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 2,
  gradient_accumulation_steps = 4,
  # eval_accumulation_steps = 8,
  evaluation_strategy = "epoch",
  predict_with_generate=True,
  generation_max_length = 128,
  # save_steps = 5000,
  save_steps = 5000,
  logging_steps = 1000,
  push_to_hub = False
)
# %%
trainer = Seq2SeqTrainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  compute_metrics = compute_metrics,
  train_dataset = tokenized["train"],
  eval_dataset = tokenized["valid"],
  tokenizer = tokenizer,
)

#%%
trainer.train()

# load from pretrained
trainer.train("result\mbart50_all_5epoch\checkpoint-495000")