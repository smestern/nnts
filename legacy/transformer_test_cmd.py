import sys
sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\")

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GPT2Config, GenerationConfig, AutoModelForSeq2SeqLM, Cohere2Config
from joblib import dump, load
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib.pyplot as plt
from nnDS import nnDS
print(torch.cuda.is_available())


def compute_metrics(eval):
    plt.clf()
    plots_n = eval.label_ids.shape[0] if eval.label_ids.shape[0] < 5 else 5
    #make a gride
    fig, axs = plt.subplots(plots_n, 1, num=1)
    #plot it lmao
    decoded_data = []
    decoded_pred = []
    for i in range(plots_n):
        decoded_data.append(data.tokenizer.decode(eval.label_ids[i]))
        decoded_pred.append(data.tokenizer.decode(eval.predictions[i].argmax(axis=-1)))
        axs[i].plot(decoded_pred[i])
        axs[i].plot(decoded_data[i])
        axs[i].set_ylim([0, 1])
    plt.pause(5)
    decoded_data = np.vstack(decoded_data)
    decoded_pred = np.vstack(decoded_pred)

    #compute mse
    mse = np.square(decoded_data - decoded_pred)
    return {'mse':mse.mean()}



DS_SAMPLE_RATE = 10000 #10kHz sample rate
#load the dataset
data = nnDS( "nn_ds_c.joblib","nn_ds.joblib", 512, 128, 0, tokenizer="tokenizer.joblib",
            length=512*10, dtype=torch.LongTensor, output_format="tokenized") 
train_set, val_set = torch.utils.data.random_split(data, [len(data)-100, 100])

#train_loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
#val_loader = DataLoader(val_set, batch_size=8)

llcfg = Cohere2Config(vocab_size=data.vocab_size, hidden_size=2048, intermediate_size=2048*2, num_attention_heads=32, num_hidden_layers=24)

model = AutoModelForCausalLM.from_config(llcfg).to("cuda")
#model = AutoModelForCausalLM.from_pretrained("C:\\Users\\SMest\\Dropbox\\nnGAN\\high_res_cohere2\\checkpoint-1255").to("cuda")
training_args = TrainingArguments(
    output_dir="high_res_cohere2",
    eval_steps=500,
    num_train_epochs=1,
    learning_rate=1e-3,
    weight_decay=0.01,
    push_to_hub=False, per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True,
    save_total_limit=4,
    optim="adamw_bnb_8bit",
    #per_device_train_batch_size=16,
    #per_device_eval_batch_size=16,
    tf32=True,
    #fp16=True,
    #use_cpu=True,


)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)

trainer.train()
plt.close()

#DUMP THE TOKENIZER
#dump(data.tokenizer, "tokenizer.joblib")
#plot the latent space
# import umap
# plt.figure(num=99)
# latent = model.transformer.wte.weight.cpu().detach().numpy()
# embed = umap.UMAP(n_neighbors=25, min_dist=0.1, metric='cosine').fit_transform(latent)
# c = ['k' if i < 2048 else 'r' for i in range(len(latent))]
# plt.scatter(embed[:, 0], embed[:, 1], color=c)
# plt.colorbar()
# plt.pause(5)

idcs = np.random.choice(len(data), 10)
#make a some noise
for i in range(10):
    plt.figure(num=i)
    #unsqueeze the data to ensure batch size 1;
    batched = {key: val.unsqueeze(0)[:, :-100].to('cuda') for key, val in data[idcs[i]].items()}
    batched.pop('token_type_ids')
    config = GenerationConfig(max_new_tokens=100, num_beams=5, pad_token_id=-1)

    pred = model.generate(**batched, generation_config=config)
    pred = [data.tokenizer.decode(x) for x in pred.cpu().numpy()]
    plt.plot(data.tokenizer.decode(data[idcs[i]]['input_ids'][:-100].cpu().numpy()), color="r")
    [plt.plot(x) for x in pred]
    
    plt.ylim([0, 2])
    plt.pause(5)
plt.show()

#
 