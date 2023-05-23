#include "cuda_runtime.h"
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <iomanip>

using namespace std;

__global__ void addVec(int N, int *arr1, int *arr2, int *arr3)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

    if (tId < N)
    {
        arr3[tId] = arr1[tId] + arr2[tId];
    }
}

void cpuAdd(int N, int arr1[], int arr2[], int arr3[])
{
    for (int i = 0; i < N; i++)
    {
        arr3[i] = arr1[i] + arr2[i];
    }
}

bool verify(int N, int *C, int *D)
{
    for (int i = 0; i < N; i++)
    {
        if (C[i] != D[i])
        {
            return false;
        }
    }

    return true;
}

int main()
{
    int N;
    clock_t start, end;
    double CPUTime, GPUTime;
    srand(time(NULL));
    cout << "Enter the value of N : ";
    cin >> N;

    int *A = new int[N];
    int *B = new int[N];
    int *C = new int[N];
    int *D = new int[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    // CPU Addition
    start = clock();
    cpuAdd(N, A, B, C);
    end = clock();
    CPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "CPUTime : " << CPUTime << endl;

    // Allocate memory to GPU
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(int) * N);
    cudaMalloc(&d_B, sizeof(int) * N);
    cudaMalloc(&d_C, sizeof(int) * N);

    // Copy Host Vector to Device
    cudaMemcpy(d_A, A, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Run the kernel
    start = clock();
    addVec<<<N / 2 + 1, 2>>>(N, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    end = clock();
    GPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "GPUTime : " << GPUTime << endl;

    // Copy Result back to CPU memory
    cudaMemcpy(D, d_C, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Verify the answer
    bool result = verify(N, C, D);

    if (!result)
    {
        cout << "Verification Failed";
    }
    else
    {
        cout << "Verification Passed!";
    }

    cout << "\nSpeedUp : " << fixed << setprecision(10) << CPUTime / GPUTime << endl;

    return 0;
}