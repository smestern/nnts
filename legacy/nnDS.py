from joblib import load
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from build_VAE_TOKEN import VanillaVAE
class nnTokenizerPasthrough():
    #tokenizer for the nn dataset, in this case it is just a passthrough really, for more complex tokenization, see below
    def __init__(self, vocab, seed=0, shuffle=True):
        #vocab is a list of unique values in the dataset
        self.vocab = vocab
        self.vocab_size = len(vocab)
        #This is a passthrough tokenizer, so the vocab is just the unique values in the dataset,
        #however this encodes tokens in numerical order, so the first token is 0, the second is 1, etc.
        #maybe this worsens performance. try to randomize the order of the tokens
        self.tokens = np.arange(self.vocab_size)

    def __call__(self, x):
        #tokenize the input
        #first bin the data
        x = self.vocab[np.digitize(x, self.vocab)-1]
        return x
    
    def decode(self, x):
        #decode the input
        return x
    
class nnTokenizer():
    #tokenizer for the nn dataset, 
    def __init__(self, vocab, tokens=None, seed=0, shuffle=True):
        #vocab is a list of unique values in the dataset
        self.vocab = vocab
        self.vocab_size = len(vocab)
        #make our tokens
        if tokens is not None:
            self.tokens = tokens
        else:
            self.tokens = np.arange(self.vocab_size)
        #shuffle? 
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.tokens)
        #create a dictionary to map the tokens to the vocab
        self.token2vocab = {self.tokens[i]: self.vocab[i] for i in range(self.vocab_size)}
        self.vocab2token = {self.vocab[i]: self.tokens[i] for i in range(self.vocab_size)}

    def __call__(self, x):
        #tokenize the input
        #first bin the data
        x = self.vocab[np.digitize(x, self.vocab)-1]
        #now encode the data
        x = np.array([self.vocab2token[i] for i in x])
        return x
    
    def decode(self, x):
        #decode the input
        x = np.array([self.token2vocab[i] for i in x])
        return x

class nnVaeTokenizer():
    def __init__(self, vocab, load='vae2.pt', tokens=None, tokenizer=nnTokenizerPasthrough, input_dim=64, latent_dim=5, seed=0, shuffle=True):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.load = load
        self.tokens = tokens
        self.seed = seed
        self.shuffle = shuffle
        self.tokenizer = nnTokenizerPasthrough(vocab, seed, shuffle)
        self.model = VanillaVAE(input_dim, latent_dim=latent_dim).to("cuda")
        self.model.load_state_dict(torch.load(load))
        self.model.eval()

    def __call__(self, x):
        x = self.tokenizer(x)
        x = torch.tensor(x).unsqueeze(0).float().to("cuda")
        #x will be in the shape of [1, N_TimeSteps]
        #we need it in [N_TimeSteps//64, 64]
        x = x.view(-1, 64)
        x = self.model.encode(x)
        #morph back to the original shape
        x = x.view(-1)
        return x.cpu().detach().numpy()
    
    def decode(self, x):
        x = torch.tensor(x).unsqueeze(0).float().to("cuda")
        x = self.model.decode(x)
        x = self.tokenizer.decode(x.cpu().detach().numpy())
        return x
    
class ElectrophysiologyTokenizer:
    def __init__(self, vocab_size=2400, spike_threshold=-20):
        self.vocab_size = vocab_size
        self.spike_threshold = spike_threshold
        self.voltage_bins = np.linspace(-120, 40, vocab_size//2)
        self.derivative_bins = np.linspace(-50, 50, vocab_size//4)
        self.special_tokens = vocab_size//4  # for spikes, bursts, etc.
    
    def encode(self, voltage_trace):
        tokens = []
        # Detect spikes and mark them specially
        spikes = self.detect_spikes(voltage_trace)
        dv_dt = np.gradient(voltage_trace)
        
        for i, (v, dv) in enumerate(zip(voltage_trace, dv_dt)):
            if spikes[i]:
                tokens.append(self.vocab_size - 1)  # Special spike token
            elif abs(dv) > 10:  # Fast transient
                tokens.append(len(self.voltage_bins) + np.digitize(dv, self.derivative_bins))
            else:  # Regular voltage
                tokens.append(np.digitize(v, self.voltage_bins))
        return tokens
    
    def detect_spikes(self, voltage_trace):
        spikes = np.zeros_like(voltage_trace, dtype=bool)
        for i in range(1, len(voltage_trace) - 1):
            if (voltage_trace[i] > self.spike_threshold and 
                voltage_trace[i] > voltage_trace[i - 1] and 
                voltage_trace[i] > voltage_trace[i + 1]):
                spikes[i] = True
        return spikes
    
    def decode(self, tokens):
        voltage_trace = []
        for token in tokens:
            if token == self.vocab_size - 1:
                voltage_trace.append(self.spike_threshold)
            elif token >= len(self.voltage_bins):
                voltage_trace.append(self.derivative_bins[token - len(self.voltage_bins)])
            else:
                voltage_trace.append(self.voltage_bins[token])
        return voltage_trace

class nnScaler():
    def __init__(self, low=None, high=None, offset=0):
        self.low = low
        self.high = high
        self.offset = offset

    def __call__(self, x):
        
        if self.low is None:
            self.low = np.min(x)
        if self.high is None:
            self.high = np.max(x)
        return np.clip((x - self.low)/(self.high - self.low), 1e-6, 0.99999) + self.offset
    
    def transform(self, x):
        return self.__call__(x)
    
    def inverse(self, x):
        x = np.clip(x - self.offset, 1e-6, 0.99999)
        return x*(self.high - self.low) + self.low
    
    def inverse_transform(self, x):
        return self.inverse(x)

class nnDS(Dataset):
    def __init__(self, stim_file, resp_file, context_length, prediction_length, max_lags,
                  length=None, transform=None, target_transform=None, neuron=None, sample_rate=10000,
                  data_scale=[-120, 40], stim_scale=[-2000,2000], tokenizer=None, lags=4, output_format='timeseries', 
                  scale=True, dtype=torch.LongTensor):
        self.stim = load(stim_file)
        self.resp = load(resp_file)
        #drop the nan
        self.stim = [x for x in self.stim if not isinstance(x, float)]
        self.resp = [x for x in self.resp if not isinstance(x, float)]
        lens = []
        self.transform = transform
        self.target_transform = target_transform
        self.max_lags = max_lags
        self.context_length = context_length + max_lags
        self.prediction_length = prediction_length
        self.window = self.context_length + self.prediction_length  # Ensure window is correctly calculated
        self.sample_rate = sample_rate
        if scale:
            #scale resp from to 0-1 using the range -120 to 120, #clip the data to -120 to 120
            self.resp_scaler = nnScaler(data_scale[0], data_scale[1])
            for i in range(len(self.resp)):
                for j in range(len(self.resp[i])):
                    lens.append(len(self.resp[i][j])/self.window)
                    self.resp[i][j] = np.nan_to_num(self.resp_scaler(self.resp[i][j]))
            self.resp_vocab = np.linspace(0, 1, 2400)

            #scale stim from to 0-1 using the range 2nA to 2nA, #clip the data to -120 to 120, data is in pA
            self.stim_scaler = nnScaler(stim_scale[0], stim_scale[1], offset=1)
            for i in range(len(self.stim)):
                for j in range(len(self.stim[i])):
                    self.stim[i][j] = np.nan_to_num(self.stim_scaler(self.stim[i][j])) 
            self.stim_vocab = np.linspace(0, 1, 2400) + 1

            self.vocab = np.hstack([self.resp_vocab, self.stim_vocab])

        else:
            self.vocab = np.linspace(-120, 120, 2400)

        self.dtype = dtype
        self.neuron = neuron
        self.lags = lags
        self.len = int(np.sum(lens)) if length is None else length

        #find the "vocab size"
        self.vocab_size = len(self.vocab)
        if tokenizer is None:
            self.tokenizer = nnTokenizer(self.vocab)
            #stim vocab is different from the response vocab
           
        else:
            #load the tokenizer if its a string
            if isinstance(tokenizer, str):
                self.tokenizer = load(tokenizer)
            #if its a class initialize it
            elif isinstance(tokenizer, type):
                self.tokenizer = tokenizer(self.vocab)
        self.output_format = output_format

    def __len__(self):
        return self.len

    def __getitem__(self, idx=None):
        if self.neuron is None and idx < len(self.stim):
            rnd_unit = idx
        elif self.neuron is not None:
            rnd_unit = self.neuron
        else:
            np.random.seed(idx)
            rnd_unit = np.random.randint(0, len(self.stim), 1)[0]
        
         
        attention_mask = torch.ones(self.window)
        
        rnd_idx_s = np.random.randint(0, len(self.stim[rnd_unit]), 1)[0]
        if len(self.resp[rnd_unit][rnd_idx_s]) < self.window:
            #if the window is larger than the data, just pad it
            rnd_idx_t = 0

            c_out = torch.tensor(self.tokenizer(self.stim[rnd_unit][rnd_idx_s])).type(self.dtype)
            y_out = torch.tensor(self.tokenizer(self.resp[rnd_unit][rnd_idx_s])).type(self.dtype)
            #pad
            c_out = torch.nn.functional.pad(c_out, (0, self.window - len(c_out)), mode="constant", value=0)
            y_out = torch.nn.functional.pad(y_out, (0, self.window - len(y_out)), mode="constant", value=0)
            x_out = torch.arange(0, self.window/self.sample_rate, step=1/self.sample_rate).type(self.dtype)
            attention_mask[:len(c_out)] = 1
        else:
            rnd_idx_t = np.random.randint(0, self.stim[rnd_unit][rnd_idx_s].shape[0]-self.window, 1)[0]
            x_out = torch.arange(0, self.window/self.sample_rate, step=1/self.sample_rate).type(self.dtype)
            c_out = torch.tensor(self.tokenizer(self.stim[rnd_unit][rnd_idx_s][rnd_idx_t:rnd_idx_t+self.window])).type(self.dtype)
            y_out = torch.tensor(self.tokenizer(self.resp[rnd_unit][rnd_idx_s][rnd_idx_t:rnd_idx_t+self.window])).type(self.dtype)
        #no need to tokenize as the data is already tokenized
        if self.output_format == 'timeseries':
            time_features = torch.cat([c_out.unsqueeze(1), x_out.unsqueeze(1)], dim=1)

            return dict(past_values=y_out[:self.context_length],
                        past_time_features=time_features[:self.context_length], 
                        past_observed_mask=attention_mask[:self.context_length], 
                        future_time_features=time_features[self.context_length:], 
                        future_values=y_out[self.context_length:],
                        future_observed_mask=attention_mask[self.context_length:], )
        elif self.output_format == 'tokenized-gpt2':
            input_ids = torch.cat([c_out, y_out[:self.context_length]])
            token_type_ids = torch.cat([torch.zeros_like(c_out), torch.ones_like(y_out[:self.context_length])])
            attention_mask = torch.cat([attention_mask[:len(c_out)], torch.ones_like(y_out[:self.context_length])])
            return dict(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=input_ids)
        elif self.output_format == 'tokenized':
            input_ids = torch.cat([c_out, y_out[:self.context_length]])
            token_type_ids = torch.cat([torch.zeros_like(c_out), torch.ones_like(y_out[:self.context_length])])
            attention_mask = torch.cat([attention_mask[:len(c_out)], torch.ones_like(y_out[:self.context_length])])
            return dict(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=input_ids)
            
        else:
            raise ValueError("Invalid output_format. Choose either 'timeseries' or 'tokenized'.")

    def get_cell(self, idx):
        return self.stim[idx], self.resp[idx]