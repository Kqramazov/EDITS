N=10
SEED=10
FILENAME='classification.py'

DATASET1='german'
DATASET2='credit'
DATASET3='bail'
DATASET4='pokec_z'
DATASET5='pokec_n'

ENCODER1='sage'
ENCODER2='gin'
ENCODER3='gcn'

# # 循环执行 Python 程序

# for i in $(seq 1 $N); do
# #   SEED=1
#   echo "Iteration $i, Seed: $SEED"
#   python train.py --dataset $DATASET1 --seed $SEED
# #   echo "$CMD --seed=$SEED 2>> $RESULT_LOG_FILE | tee -a $LOG_FILE" >> $RESULT_LOG_FILE
#   python $FILENAME  --encoder $ENCODER1 --dataset $DATASET1  --seed $SEED
#   python $FILENAME  --encoder $ENCODER2 --dataset $DATASET1  --seed $SEED
#   python $FILENAME  --encoder $ENCODER3 --dataset $DATASET1  --seed $SEED
# done


# for i in $(seq 1 $N); do
# #   SEED=1
#   echo "Iteration $i, Seed: $SEED"
#   python train.py --dataset $DATASET2 --seed $SEED
# #   echo "$CMD --seed=$SEED 2>> $RESULT_LOG_FILE | tee -a $LOG_FILE" >> $RESULT_LOG_FILE
#   python $FILENAME  --encoder $ENCODER1 --dataset $DATASET2  --seed $SEED
#   python $FILENAME  --encoder $ENCODER2 --dataset $DATASET2  --seed $SEED
#   python $FILENAME  --encoder $ENCODER3 --dataset $DATASET2  --seed $SEED
# done


# for i in $(seq 1 $N); do
# #   SEED=1
#   echo "Iteration $i, Seed: $SEED"
#   python train.py --dataset $DATASET3 --seed $SEED
# #   echo "$CMD --seed=$SEED 2>> $RESULT_LOG_FILE | tee -a $LOG_FILE" >> $RESULT_LOG_FILE
#   python $FILENAME  --encoder $ENCODER1 --dataset $DATASET3  --seed $SEED
#   python $FILENAME  --encoder $ENCODER2 --dataset $DATASET3  --seed $SEED
#   python $FILENAME  --encoder $ENCODER3 --dataset $DATASET3  --seed $SEED
# done
python train.py --dataset $DATASET4 --seed $SEED
#   echo "$CMD --seed=$SEED 2>> $RESULT_LOG_FILE | tee -a $LOG_FILE" >> $RESULT_LOG_FILE
python $FILENAME  --encoder $ENCODER1 --dataset $DATASET4  --seed $SEED


python train.py --dataset $DATASET5 --seed $SEED
#   echo "$CMD --seed=$SEED 2>> $RESULT_LOG_FILE | tee -a $LOG_FILE" >> $RESULT_LOG_FILE
python $FILENAME  --encoder $ENCODER1 --dataset $DATASET5  --seed $SEED
# python $FILENAME  --encoder $ENCODER1 --dataset $DATASET1  --seed $SEED