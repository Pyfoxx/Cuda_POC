# Cuda bruteforcing

This is a simple proof of concept to use gpu multi-core processing for bruteforce application

## TL;DR : The project is feasible

Hardware : 
CPU : I7-7700HQ
GPU : GTX 1060

```debug
== CUDA [915]  INFO -- init
leeeeeeeeeet's go
cpu mode : 0.08977989997947589s
gel time : 152.20980830001645s
var building : 152.42947879998246
starting the ol' cuda up
start copy
Copy time : 1.385747199994512s
allocation time 5.959038899978623
401712
lauch kernel
Cuda Kernel : 0.0586231000488624s
Total function : 123.07007170002908s
== CUDA [277635]  INFO -- add pending dealloc: cuMemFree_v2 2471326208 bytes
== CUDA [277657]  INFO -- dealloc: cuMemFree_v2 2471326208 bytes
== CUDA [277918]  INFO -- add pending dealloc: cuMemFree_v2 2471326208 bytes
== CUDA [277918]  INFO -- dealloc: cuMemFree_v2 2471326208 bytes
Total script time : 277.04012780002085s
== CUDA [277971]  INFO -- add pending dealloc: module_unload ? bytes
```

## Little review : 

This POC as proven interesting, even tough i can actually achieve faster data processing speed, i'm challenged with the data pre-treatment, and with the copying to the GPU memory.
But, to compare purely the processing speed, the use of the GPU is ~1.5x faster

Some heavy work will need to take place on the actual bottleneck (pre-processing, copy to GPU ram, allocation) tough
I will also need to see if the hashing on the GPU is feasible and at witch cost.

Was fun!
