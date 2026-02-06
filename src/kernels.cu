#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// CUDA_CHECK宏
#define CUDA_CHECK(expr) do { \
  cudaError_t err = expr; \
  if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error: %s (line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
  } \
} while (0)


template <class T>
__device__ T warp_reduce(T value) {
#pragma unroll
    for (size_t offset = warpSize/2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    return value;
}

template <typename T, size_t BLOCK_SIZE>
__global__ void traceKernel(const T* input, T* output, size_t n) {
  __shared__ T smem[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + tid;

  T sum = 0;
  for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
      sum += input[i];
  }

  T warp_sum = warp_reduce(sum);
  
  if(tid % warpSize == 0) {
      smem[tid/warpSize] = warp_sum;
  }
  __syncthreads();

  if(tid < warpSize) {
      T block_sum = (tid < (blockDim.x + 31)/warpSize) ? smem[tid] : T{0};
      block_sum = warp_reduce(block_sum);
      if(tid == 0) {
          atomicAdd(output, block_sum);
      }
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  size_t min_dim = (rows < cols) ? rows : cols;

  std::vector<T> h_diagonal(min_dim); // 减少拷贝数量，只拷贝对角线元素
  for (size_t i = 0; i < min_dim; ++i) {
    h_diagonal[i] = h_input[i * cols + i];
  }

  T *diagonal;
  T* d_result;
  CUDA_CHECK(cudaMalloc(&diagonal, min_dim * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

  // 数据拷贝到 Device
  CUDA_CHECK(cudaMemcpy(diagonal, h_diagonal.data(), min_dim * sizeof(T), cudaMemcpyHostToDevice));

  // 调用CUDA kernel，问题演变为求和
  dim3 blockSize = 1024;
  dim3 numBlocks = (min_dim + blockSize.x - 1) / blockSize.x;
  traceKernel<T, 1024><<<numBlocks, blockSize>>>(diagonal, d_result, min_dim);

  // 将结果从 Device 拷回 Host
  T h_result;
  CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

  // 释放 GPU 内存
  CUDA_CHECK(cudaFree(diagonal));
  CUDA_CHECK(cudaFree(d_result));

  return h_result;
}

// test
// template <typename T>
// T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
//     T sum = T{0};
//     int m = (rows < cols)? rows : cols;
//     for(size_t i = 0; i < m; ++i) {
//       if(i*cols + i < h_input.size()) {
//         sum += h_input[i*cols + i];
//       }
//     }

//     return sum;
// }

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // 没做完
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
