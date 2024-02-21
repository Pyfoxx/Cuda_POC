from numba import jit
from numba import cuda
import numpy as np
from itertools import product
from timeit import default_timer as timer

print("leeeeeeeeeet's go")
# @cuda.jit('void(ulong[:], ulong)')
# # @jit(target_backend='cuda')
# def test(to_test, expected):
#     for i in to_test:
#         if i == expected:
#             print(i)
#             break


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
    # Encode strings and target_string to numerical form here
    encoded_strings = np.array(strings)  # Placeholder for encoded input strings
    target_string_encoded = target_string  # Encoded target string
    print('start copy')
    d_encoded_strings = cuda.to_device(encoded_strings)
    print("copied")

    # Allocate memory for result (1 if match, 0 otherwise)
    result = np.zeros_like(encoded_strings)
    d_result = cuda.to_device(result)

    # Define blocks and threads for CUDA
    threads_per_block = 32
    blocks = (d_encoded_strings.size + (threads_per_block - 1)) // threads_per_block
    print(blocks)
    # Launch kernel
    print("lauch kernel")
    match_string_kernel[blocks, threads_per_block](d_encoded_strings, target_string_encoded, d_result)

print("just defined the cuda function. Nothing crazy")


def cpu_mode(exp, gen):
    for i in gen:
        if "".join(i) == exp:
            return i


expected = "abcdef"
gen = product("abcdefghijklmnopqrstuvwxyz", repeat=len(expected))

cpu_start = timer()
# cpu_mode(expected, gen)
print("--- %s seconds CPU ---" % (timer() - cpu_start))

gel = [int.from_bytes("".join(k).encode('utf-8'), 'little') for k in gen]
# print(np.array(gel, dtype=np.string_))
expected = int.from_bytes("".join(expected).encode('utf-8'), 'little')
del gen
print("starting the ol' cuda up")
start_time = timer()
griddim = 1, 2
blockdim = 3, 4
find_matching_string(gel, expected)
print("--- %s seconds GPU ---" % (timer() - start_time))