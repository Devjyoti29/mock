#include <iostream>
#include <iomanip> // for setprecision and fixed
#include <omp.h>


using namespace std;
int sum(int *arr, int n){

  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }

  return sum;
}

int parallelSum(int *arr,int n){

	int sum = 0;

  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }

  return sum;
}


int main() {
  
  int n = 10000000;
  int *arr=new int[n];
  for(int i=0;i<n;i++){
  	arr[i]=1;
  }

  double t1=omp_get_wtime();
  int sump=parallelSum(arr,n);
  double t2=omp_get_wtime();

  double ptime=t2-t1;

  t1=omp_get_wtime();
  int sums=sum(arr,n);
  t2=omp_get_wtime();

  double stime=t2-t1;

  cout<<"Parallel Sum :"<<sump<<endl;
  cout<<"Sequential Sum :"<<sums<<endl;
  cout << "Speed Up: " << fixed << setprecision(5) << stime/ptime << " seconds" << endl;
  
  return 0;
}
