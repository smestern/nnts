# %%
import sys
import os
sys.path.append("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\")
sys.path.append("C:\\Users\\SMest\\Dropbox\\scratch_space\\llm_test")
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from joblib import dump, load
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from nnDS import nnDS, nnCorrTokenizer
print(torch.cuda.is_available())


# %%

data = nnDS( "nn_ds_c.joblib","nn_ds.joblib", 512, 512, 7 )

#make a corrleation tokenizer
tokenizer = nnCorrTokenizer( n_tokens=5048, data=data,)

#replace the tokenizer in the model
data.tokenizer = tokenizer




