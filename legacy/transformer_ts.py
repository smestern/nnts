import sys
sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\")

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from joblib import dump, load
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from nnDS import nnDS
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(torch.cuda.is_available())

from nnDS import nnDS, nnTokenizerPasthrough  # Import the nnDS dataset                                                    

def compute_metrics(eval):
    plt.clf()
    plots_n = eval.label_ids.shape[0] if eval.label_ids.shape[0] < 5 else 5
    #make a gride
    fig, axs = plt.subplots(plots_n, 1, num=1)
    #plot it lmao
    
    for i in range(plots_n):
        axs[i].plot(data.tokenizer.decode(eval.predictions[i].argmax(axis=-1)))
        axs[i].plot(data.tokenizer.decode(eval.label_ids[i]))

    plt.pause(5)

    #compute mse
    mse = np.square(eval.predictions.argmax(axis=-1) - eval.label_ids)
    return {'mse':mse.mean()}

DS_SAMPLE_RATE = 10000 #10kHz sample rate

#load the dataset
data = nnDS( "nn_ds_c.joblib","nn_ds.joblib", 2500, 100, 50, 
            length=512*1000, tokenizer=nnTokenizerPasthrough, dtype=torch.float32, scale=True)  # Use nnDS dataset
train_set, val_set = torch.utils.data.random_split(data, [len(data)-100, 100])

lags_s = [0.001, 0.01, 0.1] #lags in seconds
lags = [int(x*DS_SAMPLE_RATE) for x in lags_s] #convert to samples


# llcfg = TimeSeriesTransformerConfig(context_length=int(0.1*DS_SAMPLE_RATE),
#                                      prediction_length=10, 
#                                      num_time_features=2,
#                                      #num_dynamic_real_features=1,
#                                      embedding_dimension=None,
#                                      d_model=24,
#                                      encoder_ffn_dim=512,
#                                      decoder_ffn_dim=512,
#                                      encoder_attention_heads=6,
#                                      encoder_layers=3,
#                                      lags_sequence=lags,
#                                      decoder_attention_heads=6,
#                                      decoder_layers=3,)
config = TimeSeriesTransformerConfig(
    prediction_length=100,  # Predict 2ms at 50kHz
    context_length=2500,    # 50ms context
    lags_sequence=[1, 5, 10, 25, 50],  # Multi-scale lags
    num_time_features=2,
    d_model=512,            # Increase model capacity
    encoder_layers=8,       # Deeper for complex patterns
    decoder_layers=6,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    # Add relative position encoding for better temporal modeling
    #use_relative_position_bias=True,
)

model = TimeSeriesTransformerForPrediction(config).to("cuda")
#model = TimeSeriesTransformerForPrediction.from_pretrained("C:/Users/SMest/Dropbox/nnGAN/tscool2/checkpoint-95500/").to("cuda")
training_args = TrainingArguments(
    output_dir="tscool2",
    weight_decay=0.,
    eval_steps=500,
    num_train_epochs=3,
    warmup_steps=100,
    learning_rate=1e-9,
    push_to_hub=False, #, gradient_accumulation_steps=4, gradient_checkpointing=True,
    per_device_train_batch_size=16,
    save_total_limit=3,
    tf32=True,
    max_grad_norm=0.1,
    #fp16=True,
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



#make a some noise
for i in range(10):
    plt.figure(num=i)
    y = data[np.random.randint(0, len(data))]
    

    
    pred = model.generate(past_values=y['past_values'].unsqueeze(0).to("cuda"), 
                          past_time_features=y['past_time_features'].unsqueeze(0).to("cuda"), 
                          future_time_features=y['future_time_features'].unsqueeze(0).to("cuda"),)
    pred = [data.tokenizer.decode(x) for x in pred['sequences'][0].cpu().numpy()]
    #get the distance from the future values
    future_val = data.tokenizer.decode(y['future_values'].cpu().numpy())
    dist = [np.linalg.norm(data.tokenizer.decode(x) - future_val) for x in pred]
    dist = 1-( np.array(dist) / np.max(dist))
    full_vales = np.hstack((np.hstack((y['past_values'].cpu().numpy(), future_val))))
    time = np.vstack((y['past_time_features'].squeeze(1).cpu().numpy(), y['future_time_features'].squeeze(1).cpu().numpy()))[:,1]
    [plt.plot(time, np.hstack((y['past_values'].cpu().numpy(), x)), c='k', alpha=dist[i]) for i, x in enumerate(pred)]
    
    plt.plot(time, data.tokenizer.decode(full_vales))

    #for the min dist generate the future values
    min_dist = np.argmax(dist) #argmax because the distance is inverted

    #new future values
    past_values = torch.tensor(np.hstack((y['past_values'].cpu().numpy(), pred[min_dist]))).unsqueeze(0).to("cuda")
    past_time_features = torch.tensor(np.vstack((y['past_time_features'].squeeze(1).cpu().numpy(), y['future_time_features'].squeeze(1).cpu().numpy()))).unsqueeze(0).to("cuda")
    future_time_features = torch.tensor(y['future_time_features'].cpu().numpy()).unsqueeze(0).to("cuda")
    future_values = model.generate(past_values=past_values[:,10:], past_time_features=past_time_features[:,10:], future_time_features=future_time_features)
    future_values = [data.tokenizer.decode(x) for x in future_values['sequences'][0].cpu().numpy()]
    #plot 
    new_time = np.linspace(time[-1], time[-1]+len(future_values[0])/DS_SAMPLE_RATE, len(future_values[0]))
    [plt.plot(np.hstack((time, new_time)), np.hstack((past_values[0].cpu().numpy(), x)), c='r') for x in future_values]

    #plt.ylim([-90, 50])
    plt.pause(5)
plt.show()

#
 