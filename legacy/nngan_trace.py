import pandas as pd
import numpy as np
import os
from joblib import dump, load, Parallel, delayed
import glob
from scipy.signal import resample
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor
from allensdk.core.cell_types_cache import CellTypesCache
path = '/media/smestern/Expansion/dandi/000020/'
ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
ctc = CellTypesCache(manifest_file='/media/smestern/Expansion/dandi/000020/manifest.json')

def downsample_ar(a, size=5000, method='resample'):
    if method == 'resample':
        new = resample(a, size)
    elif method == 'mean':
        new = np.mean([a[i:i + len(a) // size] for i in range(0, len(a), len(a) //
                        size)], axis=1)
    elif method == 'max':
        new = np.max([a[i:i + len(a) // size] for i in range(0, len(a), len(a) //
                       size)], axis=1)
    elif method == 'min':
        new = np.min([a[i:i + len(a) // size] for i in range(0, len(a), len(a) //
                       size)], axis=1)
    elif method == 'skip':
        new = a[::len(a) // size]
    return new

def round(x, base=5):
    return base * np.round(x/base)

def trim_sweep(x, y):
    diff = np.diff(x)
    _non_zero  = np.where(diff != 0)[0]
    first_non_zero = np.clip(_non_zero[1] - 1000, 0, len(x))
    last_non_zero = np.clip(_non_zero[-1] + 2000,0, len(x))
    try:
        return x[int(first_non_zero):int(last_non_zero)], y[int(first_non_zero):int(last_non_zero)]
    except: 
        return np.nan, np.nan

def load_data(file_p, ef_df, downsample_rate=10000):
    
    dir_ = os.path.dirname(file_p)

    specimen_id = os.path.basename(file_p)
    specimen_features = ef_df[ef_df['specimen_id'] == specimen_id]
    specimen_features_to_nm = specimen_features.to_numpy()[:,1:5].astype(float)
    
    if specimen_features.empty or np.all(np.isnan(specimen_features_to_nm)):
        return np.nan, np.nan
    #sweep_info = ju.read(dir_ + "\\ephys_sweeps.json")
    data_set = create_ephys_data_set(file_p)
    iclamp_st_raw = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,)# stimuli=('Long Square', 'Long Square Threshold', 'Long Square SupraThreshold', 'Long Square SubThreshold'))
    #drop sweeps with short in the stimulus_name
    drop_stim_names = ["Short", "Test", "Search"]
    for name in drop_stim_names:
        iclamp_st_raw = iclamp_st_raw[~iclamp_st_raw['stimulus_name'].str.contains(name)]
    #print(iclamp_st_raw['stimulus_name'])
    iclamp_st = iclamp_st_raw["sweep_number"].sort_values().values
    if len(iclamp_st) < 1:
        return np.nan, np.nan
    sweep_y = []
    sweep_c = []
    stimuli_name = []
    for sweep in iclamp_st:
        try:
            sweep_data = data_set.get_sweep_data(sweep)
            #stimuli_name.append(iclamp_st_raw[iclamp_st_raw['sweep_number'] == sweep]['stimulus_name'].values[0])
            mv_max = sweep_data['response'].max()
            trimmed_stim, trimmed_res = trim_sweep(sweep_data['stimulus'], sweep_data['response'])
            #downsample to 10000hz
            dwn_len = int((len(trimmed_res) / sweep_data['sampling_rate']) * downsample_rate)
            if (sweep_data['stimulus'].min() < 0) == False and (mv_max  > -120 and mv_max < 120) and dwn_len >= 512:
                dwnsampled_r = downsample_ar(trimmed_res, dwn_len, method='skip') 
                dwnsampled_c = downsample_ar(trimmed_stim, dwn_len, method='skip') #* 10e11 #convert to pA
                #dwnsampled_r = round(dwnsampled, 0.1)
                #dwnsampled_c = round(dwnsampled_c,0.1)
                if np.any(np.isnan(dwnsampled_r)):
                    continue
                sweep_c.append(dwnsampled_c )
                sweep_y.append(dwnsampled_r)
                stimuli_name.append(iclamp_st_raw[iclamp_st_raw['sweep_number'] == sweep]['stimulus_name'].values[0])
        except:
            continue
    if len(sweep_y) < 2:
       return np.nan, np.nan #raise Exception("Not enough sweeps")
    return sweep_y, sweep_c, dict(specimen_features.iloc[0])


def nwb_pass_thru(file, ef_df):
    try:
        return load_data(file, ef_df)
    except:
        return np.nan, np.nan



def gen_dataset(num=2000):
    nwbs = glob.glob(path + "/**/*.nwb", recursive=True)

    
    ef_df = pd.read_csv('/media/smestern/Expansion/dandi/000020.csv')

    ef_df["specimen_id"] = [os.path.basename(x) for x in ef_df["specimen_id"]]

    ds_len = len(nwbs)
    if num > ds_len:
        num = ds_len
        print(f"Requested dataset size larger than available NWBs, setting to max size {ds_len}")
    units_y = []
    units_c = []
    np.random.seed(0)
    res = Parallel(n_jobs=12, verbose=5)(delayed(nwb_pass_thru)(nwbs[i], ef_df) for i in range(num))
    units_y = [i[0] for i in res if i[0] is not np.nan]
    units_c = [i[1] for i in res if i[1] is not np.nan]
    #units_y_stack = np.vstack(units_y)
    #units_c_stack = np.vstack(units_c)
    dump(units_y, "nn_ds.joblib", compress=9)
    dump(units_c, "nn_ds_c.joblib", compress=9)
    return units_y


def main():
    units_y_stack = gen_dataset()
    
if __name__ == "__main__":
    main()
