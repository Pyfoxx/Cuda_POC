from numba import jit
from numba import cuda
import numpy as np
from itertools import product
from timeit import default_timer as timer

print("leeeeeeeeeet's go")
total_time = timer()


@cuda.jit('void(ulong[:], ulong, float32[:])')
def match_string_kernel(encoded_strings, target_string_encoded, result):
    # This is a placeholder function body. You'll need to adapt it based on how you encode your strings.
    # threadIdx.x is the unique thread index within the block
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= encoded_strings.shape[0]: return  # Check if idx is beyond the array length

    # Example comparison logic (for numerical representation of strings)
    if encoded_strings[idx] == target_string_encoded:
        result[idx] = 1  # Mark as match

def find_matching_string(strings, target_string):
    func_start_time = timer()
    # Encode strings and target_string to numerical form here
    encoded_strings = np.array(strings)  # Placeholder for encoded input strings
    target_string_encoded = target_string  # Encoded target string
    print('start copy')
    copy_start_time = timer()
    d_encoded_strings = cuda.to_device(encoded_strings)
    print(f"Copy time : {timer() - copy_start_time}s")
    del copy_start_time

    # Allocate memory for result (1 if match, 0 otherwise)
    alloc_time = timer()
    result = np.zeros_like(encoded_strings)
    d_result = cuda.to_device(result)
    print(f"allocation time {timer() - alloc_time}")
    del alloc_time

    # Define blocks and threads for CUDA
    threads_per_block = 769
    blocks = (d_encoded_strings.size + (threads_per_block - 1)) // threads_per_block
    print(blocks)
    # Launch kernel
    print("lauch kernel")
    cuda_start_time = timer()
    match_string_kernel[blocks, threads_per_block](d_encoded_strings, target_string_encoded, d_result)
    print(f"Cuda Kernel : {timer() - cuda_start_time}s")
    print(f"Total function : {timer()-func_start_time}s")



def cpu_mode(exp, gen):
    for i in gen:
        if "".join(i) == exp:
            return i

var_time = timer()
expected = "abcdef"
gen = product("abcdefghijklmnopqrstuvwxyz", repeat=len(expected))
cpu_time = timer()
cpu_mode(expected, gen)
print(f"cpu mode : {timer() - cpu_time}s")
gen = product("abcdefghijklmnopqrstuvwxyz", repeat=len(expected))
gel_time = timer()
gel = [int.from_bytes("".join(k).encode('utf-8'), 'little') for k in gen]
print(f"gel time : {timer()-gel_time}s")
# print(np.array(gel, dtype=np.string_))
expected = int.from_bytes("".join(expected).encode('utf-8'), 'little')
print(f"var building : {timer()-var_time}")
del gen, var_time
print("starting the ol' cuda up")
griddim = 1, 2
blockdim = 3, 4
find_matching_string(gel, expected)
print(f"Total script time : {timer() - total_time}s")