## Weight Parallel Unit Launch Experiment 

## Contents 
基于 weigits2excel_git, 进行权重并行单元发射实验. 

目标权重：/home/aiden00/abwu_workspace/yolov4best/yolov4.best.0 
目标配置文件：/home/aiden00/abwu_workspace/yolov4best/yolov4.cfg 　
V4-prune9  Map: 82.71％  IOU: 77.11%  Sparsity: 93.34%  v14(add pl2) 

以第157层：1024*512*3*3为例, 
切片管理：每次切4片IC, 对OC分64段, 共64个CU, 每个CU4个DSP, 共256个DSP, 
以第0, 1, 2, 3层IC为例，　第0, 16, 32, ...共64个OC同时发射, 
记录64个CU中每个CU, 全零发射次数的[min, max], 
以及非全零即正常发射次数的max, 共发射16(1024/64)波, 对max进行累加. 
遍历所有IC最后统计发射占比. 

通过运行weights_parallel_all.py，weights_parallel_layer.py, 
得到输出日志'133_3.log'等等, 

通过运行parallel_all2excell.py, 
保存至output文件夹的excell文件. 
