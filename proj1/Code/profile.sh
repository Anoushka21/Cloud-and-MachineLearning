#!/bin/bash

MODEL=$1
GPU=$2
NCU_PATH="/share/apps/cuda/11.1.74/bin/ncu"
NSYS_PATH="/share/apps/cuda/11.1.74/bin/nsys"

CMD="python3 main2.py -a $MODEL --gpu 0 --epoch 1 -b 16 /scratch/ag8733/CML/proj1/imagenet-mini"
METRICS="dram__sectors.sum,dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"

rm -rf $GPU/$MODEL
mkdir $GPU/$MODEL
NCU_RAW_LOG="$GPU/$MODEL/ncu-log-$MODEL.txt"
NCU_METRICS_LOG="$GPU/$MODEL/ncu-final-metric-$MODEL.txt"

echo "********** NCU PROFILING STARTS **********"

$NCU_PATH  --profile-from-start off -f --log-file $NCU_RAW_LOG --metrics $METRICS --target-processes all $CMD

# compute bytes
BYTES=`cat $NCU_RAW_LOG | grep -e "dram__bytes.sum" | grep -e " byte" | sed -e "s/,/ /g" | awk '{print($3)}' | paste -sd+ | bc`
KBYTES=`cat $NCU_RAW_LOG | grep -e "dram__bytes.sum" | grep -e "Kbyte" | sed -e "s/,/ /g"  |awk '{print($3)}' | paste -sd+ | bc`
MBYTES=`cat $NCU_RAW_LOG | grep -e "dram__bytes.sum" | grep -e "Mbyte" | sed -e "s/,/ /g" | awk '{print($3)}' | paste -sd+ | bc`
GBYTES=`cat $NCU_RAW_LOG | grep -e "dram__bytes.sum" | grep -e "Gbyte" | sed -e "s/,/ /g" | awk '{print($3)}' | paste -sd+ | bc`
DEFAULT_BYTES=0
DEFAULT_KBYTES=0
DEFAULT_MBYTES=0
DEFAULT_GBYTES=0
# Assign default values if variables are empty
BYTES=${BYTES:-$DEFAULT_BYTES}
KBYTES=${KBYTES:-$DEFAULT_KBYTES}
MBYTES=${MBYTES:-$DEFAULT_MBYTES}
GBYTES=${GBYTES:-$DEFAULT_GBYTES}
TOTAL_BYTES=$(awk "BEGIN{ print $BYTES + 1000*$KBYTES + 1000000*$MBYTES + 1000000000*$GBYTES }")
echo "TOTALBYTES" >> $NCU_METRICS_LOG
echo $TOTAL_BYTES >> $NCU_METRICS_LOG

#compute read write
DRW=`cat $NCU_RAW_LOG | grep -e "dram__sectors.sum" | sed -e "s/,//g" |   awk '{print($3)}' | paste -sd+ | bc`
echo $'\nDRDW TRANSACTIONS' >> $NCU_METRICS_LOG
echo $DRW >> $NCU_METRICS_LOG

#compute FLOPS
TMP1=`cat $NCU_RAW_LOG | grep -e "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum" -e  "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum" | sed -e "s/,//g" | awk '{print $3}' | paste -sd+ | bc`
TMP2=`cat $NCU_RAW_LOG | grep -e "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum" | sed -e "s/,//g" | awk '{print $3}' | paste -sd+ | bc`
TOTALFLOPS=$((TMP1 + 2*TMP2))
echo "\nTOTALFLOPS" >> $NCU_METRICS_LOG
echo $TOTALFLOPS >> $NCU_METRICS_LOG

echo $'********** NCU PROFILING ENDED **********\n'


# nsys file
NSYS_RAW_LOG="$GPU/$MODEL/nsys-log-$MODEL.qdrep"
NSYS_METRICS_LOG="$GPU/$MODEL/nsys-final-metric-$MODEL.txt"

echo "********** NSYS PROFILING STARTS **********"
$NSYS_PATH profile -f true -o $NSYS_RAW_LOG $CMD
$NSYS_PATH stats --report gputrace $NSYS_RAW_LOG -o $GPU/$MODEL/tmp_nsys
# compute runtime
echo "Runtime(nanosec)" >> $NSYS_METRICS_LOG
tail -n +2 $GPU/$MODEL/tmp_nsys_gputrace.csv | sed -e "s/,/ /g" | awk '{print $2}' | paste -sd+ | bc >> $NSYS_METRICS_LOG
echo "********** NSYS PROFILING ENDED **********"


