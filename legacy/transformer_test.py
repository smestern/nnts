import sys
sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\")

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from joblib import dump, load
import numpy as np
from torch.utils.data import Dataset
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
from nnDS import nnDS
   

def compute_metrics(eval):
    #plot it lmao
    plt.clf()
    plt.plot(data.tokenizer.decode(eval.predictions[1]))
    plt.plot(data.tokenizer.decode(eval.label_ids[1]))
    plt.pause(5)

    #compute mse
    #mse = np.square(eval.predictions - eval.label_ids)
    return {'mse':0}

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", max_length=513)

#load the dataset
data = nnDS( "nn_ds_c.joblib","nn_ds.joblib", 256, 256, 1,
            length=512*100, output_format="tokenized")
train_set, val_set = torch.utils.data.random_split(data, [len(data)-100, 100])
training_args = Seq2SeqTrainingArguments(
    output_dir="high_res2",
    learning_rate=0.5e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.0,
    save_total_limit=3,
    num_train_epochs=15,
    predict_with_generate=True,
    tf32=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)
trainer.train()