from joblib import load, dump

#load the two datasets
dataset1 = load("nn_ds_c.joblib")
dataset2 = load("nn_ds.joblib")
obj = {'voltages': dataset2, 'currents': dataset1}
dump(obj, "nn_ds_combined.joblib", compress=3)