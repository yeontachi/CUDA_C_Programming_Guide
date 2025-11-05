// vec_add.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(err));                                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

// GPU에서 실행될 함수 (커널)
__global__ void VecAdd(const float* A, const float* B, float* C, int N){
    int i = threadIdx.x;  // 블록 1개 가정(예제 구성)
    if(i<N){              // 경계 검사
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    //----------parameters----------
    const int N = 256;
    const size_t BYTES = N * sizeof(float);

    //----------host memory allocation----------
    std::vector<float> hA(N), hB(N), hC(N), hRef(N);

    // 입력 초기화
    for(int i=0; i<N; ++i){
        hA[i] = 0.5f * i;
        hB[i] = 2.0f * i + 1.0f;
        hRef[i] = hA[i] + hB[i]; // cpu 기준 정답
    }

    //----------device memory allocation----------
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, BYTES));
    CUDA_CHECK(cudaMalloc(&dB, BYTES));
    CUDA_CHECK(cudaMalloc(&dC, BYTES));

    //----------copy data from host to device----------
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), BYTES, cudaMemcpyHostToDevice));

    //----------launch kernel----------
    dim3 grid(1);       // 블록 1개
    dim3 block(N);      // 스레드 N개
    VecAdd<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError()); // 커널 실행 에러 체크
    CUDA_CHECK(cudaDeviceSynchronize()); // 커널 완료 대기

    //----------copy data from device to host----------
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, BYTES, cudaMemcpyDeviceToHost));

    //----------verify results----------
    int mismatch = -1;
    for(int i=0; i<N; ++i){
        if(std::fabs(hRef[i] - hC[i]) > 1e-6f){
            mismatch = i;
            break;
        }
    }
    if (mismatch < 0) {
        printf("[OK] GPU 결과가 CPU 기준과 일치합니다. (N=%d)\n", N);
    } else {
        printf("[FAIL] mismatch at %d: gpu=%f, cpu=%f\n",
               mismatch, hC[mismatch], hRef[mismatch]);
    }

    // 앞부분 일부만 보기 좋게 출력
    const int SHOW = 10;
    printf("\nIdx   A[i]      B[i]      C[i]=A[i]+B[i]\n");
    for (int i = 0; i < SHOW && i < N; ++i) {
        printf("%3d  %8.3f  %8.3f  %12.3f\n", i, hA[i], hB[i], hC[i]);
    }

    // ----------free device memory----------
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}