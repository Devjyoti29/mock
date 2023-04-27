#include <iostream>
#include <iomanip> // for setprecision and fixed
#include <omp.h>


using namespace std;


int main() {
  
  int n = 10000000;
  int *arr=new int[n];
  for(int i=0;i<n;i++){
  	arr[i]=1;
  }



  int sum = 0;
  double t1=omp_get_wtime();

  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }
  double t2=omp_get_wtime();

  cout<<sum;
  cout << "Time taken: " << fixed << setprecision(5) << t2 - t1 << " seconds" << endl;
  
  return 0;
}
