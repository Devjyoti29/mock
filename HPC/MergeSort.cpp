#include <iostream>

#include<omp.h>

#include <iomanip>





using namespace std;





void merge(int arr[], int l, int m, int r)

{

    int n1 = m - l + 1;

    int n2 = r - m;



    // Create temporary arrays

    int L[n1], R[n2];



    // Copy data to temporary arrays

    for (int i = 0; i < n1; i++)

        L[i] = arr[l + i];

    for (int j = 0; j < n2; j++)

        R[j] = arr[m + 1 + j];



    // Merge the temporary arrays back into arr[l..r]

    int i = 0; // Initial index of first subarray

    int j = 0; // Initial index of second subarray

    int k = l; // Initial index of merged subarray

    while (i < n1 && j < n2) {

        if (L[i] <= R[j]) {

            arr[k] = L[i];

            i++;

        }

        else {

            arr[k] = R[j];

            j++;

        }

        k++;

    }



    // Copy the remaining elements of L[], if there are any

    while (i < n1) {

        arr[k] = L[i];

        i++;

        k++;

    }



    // Copy the remaining elements of R[], if there are any

    while (j < n2) {

        arr[k] = R[j];

        j++;

        k++;

    }

}



void parallelMergeSort(int *arr,int low, int high){

	int mid;



	if(low<high){



		mid=(low+high)/2;



		#pragma omp task shared(arr) if (high-low > 100)

		parallelMergeSort(arr,low,mid);



		#pragma omp task shared(arr) if (high-low > 100)

		parallelMergeSort(arr,mid+1,low);





		#pragma omp taskwait

		merge(arr,low,mid,high);



	}

	

}







void mergeSort(int *arr,int low, int high){

	int mid;



	if(low<high){



		mid=(low+high)/2;

		mergeSort(arr,low,mid);

		mergeSort(arr,mid+1,low);



		merge(arr,low,mid,high);



	}

	

}







int main(){

	int N=10000,i;

	int *a=new int[N];

	int *b=new int[N];

	double t1,t2,t3,t4;

	for(i=N-1;i>=0;i--)

	{

		a[i]=i;

		b[i]=i;

	}

	t1=omp_get_wtime();

	

    
    #pragma omp parallel
	#pragma omp single
	parallelMergeSort(a,0,N-1);

	

	

        t2=omp_get_wtime();

        cout << "Time taken for parallel: " << fixed << setprecision(5) << t2 - t1 << " seconds" << endl;

        

        

        t3=omp_get_wtime();

        mergeSort(b,0,N-1);

        t4=omp_get_wtime();

        cout << "Time taken for sequential: " << fixed << setprecision(5) << t4 - t3 << " seconds" << endl;

        

}

