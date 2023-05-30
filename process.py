import pandas as pd
import numpy as np

file_name = "edits.csv"
data = pd.read_csv(file_name)
stat = data.groupby(['dataset','encoder'])

for i in stat:
    print(i[0])
    print('auc_roc:', round(np.mean(i[1]["auc_roc"]), 2), '±', round(
        np.std(i[1]["auc_roc"]), 2), sep='')
    print('f1:', round(np.mean(i[1]["f1"]),2), '±' ,round(np.std(i[1]["f1"]),2), sep='')
    print('Acc:', round(np.mean(i[1]["Acc"]),2), '±' ,round(np.std(i[1]["Acc"]),2), sep='')
    print('parity:', round(np.mean(i[1]["parity"]),2), '±', round(np.std(i[1]["parity"]),2), sep='')
    print('equality:', round(np.mean(i[1]["equality"]),2), '±', round(np.std(i[1]["equality"]),2), sep='')