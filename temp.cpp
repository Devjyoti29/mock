#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// A class to represent an undirected graph using adjacency list representation
class Graph {
    int V; // number of vertices
    vector<vector<int>> adj; // adjacency list
public:
    Graph(int V) : V(V) {
        adj.resize(V);
    }

    // function to add an edge to the graph
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // function to perform BFS starting from a given source vertex
    void BFS(int src) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[src] = true;
        q.push(src);

        while (!q.empty()) {
            // get the next vertex from the queue
            int u = q.front();
            q.pop();
	
		#pragma omp critical
            // process all neighbors of the current vertex
            #pragma omp parallel for
            for (int i = 0; i < adj[u].size(); i++) {
                int v = adj[u][i];

                // if the neighbor has not been visited yet, mark it as visited and add it to the queue
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
    }
};

int main(){
Graph g(1000);

// Add edges
for (int i = 0; i < 1000; i++) {
    for (int j = i+1; j < 1000; j++) {
        if ((i+3*j) % 7 == 0 || (j+2*i) % 5 == 0) {
            g.addEdge(i, j);
        }
    }
}


    // perform BFS starting from vertex 0
    cout << "BFS starting from vertex 0:" << endl;
    double t1,t2;

    //t1=omp_get_wtime();
    g.BFS(0);
    //t2=omp_get_wtime();
     // cout << "Time taken: " << fixed << setprecision(5) << t2 - t1 << " seconds" << endl;
}