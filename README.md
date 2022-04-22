# EDITS
Open source code for "EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks".

## Environment
Experiments are carried out on a Titan RTX with Cuda 10.1. 

Library details can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
Default dataset for node classification is bail. Pre-processed datasets (default for bail) are provided in *pre_processed*.
Use as
```
python train.py
```
for preprocessing and 
```
python classification.py
```
for downstream node classification task.

## Log example for node classification on German

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```

(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.17086205277713312
Average of all Wasserstein distance value across feature dimensions: 0.006328224176930857
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.28068607580843574
Average of all Wasserstein distance value across feature dimensions: 0.01039578058549762
****************************************************************************
100%|██████████| 500/500 [00:18<00:00, 26.48it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.05468372589251738
Average of all Wasserstein distance value across feature dimensions: 0.0023775532996746693
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.09379225766248622
Average of all Wasserstein distance value across feature dimensions: 0.004077924246195052
****************************************************************************
100%|██████████| 1000/1000 [00:16<00:00, 60.58it/s]
Optimization Finished!
Total time elapsed: 16.5090s
Delta_{SP}: 0.0008183306055645767
Delta_{EO}: 0.0010504201680672232
F1: 0.8116710875331565
AUC: 0.7140571428571428
```

## Log example for node classification on Credit

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```

(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.03643643322694976
Average of all Wasserstein distance value across feature dimensions: 0.0028028025559192126
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.06161969755359447
Average of all Wasserstein distance value across feature dimensions: 0.004739976734891882
****************************************************************************
100%|██████████| 500/500 [05:36<00:00,  1.49it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.02081495033963641
Average of all Wasserstein distance value across feature dimensions: 0.0023127722599596014
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.035380875298436004
Average of all Wasserstein distance value across feature dimensions: 0.003931208366492889
****************************************************************************
100%|██████████| 1000/1000 [04:54<00:00,  3.39it/s]
Optimization Finished!
Total time elapsed: 294.6079s
Delta_{SP}: 0.10104353944645594
Delta_{EO}: 0.07581226764553184
F1: 0.8187980945401246
AUC: 0.7305588243155289
```

## Log example for node classification on Bail

(1) Directly do node classification task with the orginal attributed network (set args.preprocessed_using as 0):
```
python classification.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.017159402967899848
Average of all Wasserstein distance value across feature dimensions: 0.0009533001648833249
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.01976526986139568
Average of all Wasserstein distance value across feature dimensions: 0.0010980705478553154
****************************************************************************
100%|██████████| 1000/1000 [01:23<00:00, 12.02it/s]
Optimization Finished!
Total time elapsed: 83.2267s
Delta_{SP}: 0.0820449428186098
Delta_{EO}: 0.0566079463128194
F1: 0.7816419612314709
AUC: 0.8678896786694952
```

(2-1) Debiasing attributed network with EDITS:
```
python train.py
```
```
****************************Before debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.017159402967899848
Average of all Wasserstein distance value across feature dimensions: 0.0009533001648833249
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.01976526986139568
Average of all Wasserstein distance value across feature dimensions: 0.0010980705478553154
****************************************************************************
100%|██████████| 100/100 [00:34<00:00,  2.93it/s]
Preprocessed datasets saved.
```
(2-2) Carry out node classification task with the output of EDITS (set args.preprocessed_using as 1):
```
python classification.py
```
```
****************************After debiasing****************************
Attribute bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.006414602640346333
Average of all Wasserstein distance value across feature dimensions: 0.000458185902881881
Structural bias : 
Sum of all Wasserstein distance value across feature dimensions: 0.008129858181420972
Average of all Wasserstein distance value across feature dimensions: 0.0005807041558157836
****************************************************************************
100%|██████████| 1000/1000 [01:17<00:00, 12.92it/s]
Optimization Finished!
Total time elapsed: 77.3908s
Delta_{SP}: 0.05741161830379604
Delta_{EO}: 0.03746663616258683
F1: 0.7984610831606985
AUC: 0.8914640940634824
```
