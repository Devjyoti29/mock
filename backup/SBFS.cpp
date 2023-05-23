#include <iostream>
#include <vector>
#include<queue>
#include<omp.h>
#include <iomanip> // for setprecision and fixed

using namespace std;

class Graph{
    int V;
    vector<vector<int>> adj;
    
    public: Graph(int V){
        this->V=V;
        adj.resize(V);
    }

    public: void addEdge(int u,int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    public: void BFS(int src){
        vector<bool> visited(V,false);
        queue<int> frontier;
        queue<int> nextFrontier;

        visited[src]=true;
        frontier.push(src);

        while(!frontier.empty()){
            for(int i=0;i<frontier.size();i++){
                int u=frontier.front();
                frontier.pop();
                cout<< u << " ";
                for(int j=0;j<adj[u].size();j++){
                    int v=adj[u][j];
                    if(!visited[v]){
                        nextFrontier.push(v);
                        visited[v]=true;
                    }
                }

                
            }
            frontier=nextFrontier;
            nextFrontier=queue<int>();
        

        }
    }

};
int main(){
	int N=30000;
Graph g(N);

// Add edges
for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
        if ((i+3*j) % 7 == 0 || (j+2*i) % 5 == 0) {
            g.addEdge(i, j);
        }
    }
}


    // perform BFS starting from vertex 0
    cout << "BFS starting from vertex 0:" << endl;
    double t1,t2;

    t1=omp_get_wtime();
    g.BFS(0);
    t2=omp_get_wtime();
      cout << "Time taken: " << fixed << setprecision(5) << t2 - t1 << " seconds" << endl;
}