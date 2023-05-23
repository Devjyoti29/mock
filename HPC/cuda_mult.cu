#include <iostream>
#include "cuda_runtime.h"
#include <time.h>

using namespace std;

__global__ void gpuMult(int *d_A, int *d_B, int *d_C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;

    if (row < m && col < k)
    {
        for (int i = 0; i < n; i++)
        {
            temp += d_A[n * row + i] * d_B[k * i + col];
        }
        d_C[k * row + col] = temp;
    }
}

void cpuMult(int *A, int *B, int *C, int m, int n, int k)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < k; col++)
        {
            int temp = 0;
            for (int i = 0; i < n; i++)
            {
                temp += A[n * row + i] * B[k * i + col];
            }
            C[k * row + col] = temp;
        }
    }
}

bool verify(int m, int k, int *C, int *D)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < k; col++)
        {

            if (C[k * row + col] != D[k * row + col])
            {
                return false;
            }
        }
    }
    return true;
}

int main()
{
    clock_t start, end;
    int m, n, k;
    cout << "Enter m, n, k : ";
    cin >> m >> n >> k;

    int *A = new int[m * n];
    int *B = new int[n * k];
    int *C = new int[m * k];
    int *D = new int[m * k];

    // Randomly initilize the matrix A
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = rand() % 1000;
        }
    }

    // Randomly initilize the matrix B
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            B[i * k + j] = rand() % 1000;
        }
    }

    // CPU Matrix Mult
    start = clock();
    cpuMult(A, B, C, m, n, k);
    end = clock();

    double CPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "CPU Time : " << CPUTime << endl;

    // Allocate memory in GPU
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(int) * m * n);
    cudaMalloc(&d_B, sizeof(int) * n * k);
    cudaMalloc(&d_C, sizeof(int) * m * k);

    // Copy Host Matrix to Device
    cudaMemcpy(d_A, A, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    // Run the kernel
    int NUM_THREADS = 32;
    int block_row = (m + NUM_THREADS - 1) / NUM_THREADS;
    int block_col = (k + NUM_THREADS - 1) / NUM_THREADS;

    dim3 BLOCKS(block_col, block_row);
    dim3 THREAD(NUM_THREADS, NUM_THREADS);

    start = clock();
    gpuMult<<<BLOCKS, THREAD>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    end = clock();
    double GPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "GPU Time : " << GPUTime << endl;

    // Copy Result back to Host Memory
    cudaMemcpy(D, d_C, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    // validate results computed by GPU
    bool result = verify(m, k, C, D);
    if (!result)
    {
        cout << "Verification Failed" << endl;
    }
    else
    {
        cout << "Verification Passed" << endl;
    }

    cout << "SpeedUp : " << CPUTime / GPUTime << endl;

    // Free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}