#include<iostream>
#include<vector>
#include <omp.h>
#include <iomanip>
#include<queue>
#include<stack>
using namespace std;

class Graph{
public:
	int V;
	vector<vector<int>> graph;
	
	Graph(int v){
		V = v;
		graph.resize(v);
	}
	
	void addEdge(int i, int j){
		graph[i].push_back(j);
		graph[j].push_back(i);
	}
	
	void bfs(int src){
		queue<int> q;
		q.push(src);
		int *visited = new int[V];
		visited[src] = 1;
		
		while(!q.empty()){
			int node = q.front();
			q.pop();
			
						
			for(int i = 0; i<graph[node].size(); i++){
				int ele = graph[node][i];
				
				int visited_ele = 0;
				
				visited_ele = visited[ele];
				
				if(visited_ele == 0){
					visited[ele] = 1;
					q.push(ele);
				}
			}
		}
	}
	
	void dfs(int src){
		stack<int> q;
		q.push(src);
		int *visited = new int[V];
		visited[src] = 1;
		
		while(!q.empty()){
			int node = q.top();
			q.pop();
			
						
			for(int i = 0; i<graph[node].size(); i++){
				int ele = graph[node][i];
				
				int visited_ele = 0;
				
				visited_ele = visited[ele];
				
				if(visited_ele == 0){
					visited[ele] = 1;
					q.push(ele);
				}
			}
		}
	}
	
	void pBfs(int src){
		queue<int> q;
		q.push(src);
		int *visited = new int[V];
		visited[src] = 1;
		
		while(!q.empty()){
			int node = q.front();
			q.pop();
			
			#pragma omp parallel for			
			for(int i = 0; i<graph[node].size(); i++){
				int ele = graph[node][i];
				
				int visited_ele = 0;
				
				#pragma omp atomic read
				visited_ele = visited[ele];
				
				if(visited_ele == 0){
				
					#pragma omp atomic write
					visited[ele] = 1;
					
					#pragma omp critical
					q.push(ele);
				}
			}
		}
	}
	
	void pDfs(int src){
		stack<int> q;
		q.push(src);
		int *visited = new int[V];
		visited[src] = 1;
		
		while(!q.empty()){
			int node = q.top();
			q.pop();
			
			#pragma omp parallel for			
			for(int i = 0; i<graph[node].size(); i++){
				int ele = graph[node][i];
				
				int visited_ele = 0;
				
				#pragma omp atomic read
				visited_ele = visited[ele];
				
				if(visited_ele == 0){
				
					#pragma omp atomic write
					visited[ele] = 1;
					
					#pragma omp critical
					q.push(ele);
				}
			}
		}
	}

};


int main(){
	int N;
	cout << "No. of Nodes : ";
	cin >> N;
	
	Graph g(N);
	
	// Randomly add Edges
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			if( (i*3+j)%7 == 0 || (i+2*j)%5 == 0){
				g.addEdge(i, j);
			}
		}
	}
	
	// Declare Timer
	double start, end;
	
	// Serial BFS
	start = omp_get_wtime();
	g.bfs(0);
	end = omp_get_wtime();
	double sBFS = end-start;
	cout << "Serial BFS Time : " << fixed << setprecision(8) << sBFS << " sec" << endl;
	
	// Parallel BFS
	start = omp_get_wtime();
	g.pBfs(0);
	end = omp_get_wtime();
	double pBFS = end-start;
	cout << "Parallel BFS Time : " << fixed << setprecision(8) << pBFS << " sec" << endl;
	
	cout << "SpeedUp of BFS : " << sBFS/pBFS << endl;
	
	// Serial DFS
	start = omp_get_wtime();
	g.dfs(0);
	end = omp_get_wtime();
	double sDFS = end-start;
	cout << "\nSerial DFS Time : " << fixed << setprecision(8) << sDFS << " sec" << endl;
	
	// Parallel DFS
	start = omp_get_wtime();
	g.pDfs(0);
	end = omp_get_wtime();
	double pDFS = end-start;
	cout << "Parallel DFS Time : " << fixed << setprecision(8) << pDFS << " sec" << endl;
	
	cout << "SpeedUp of DFS : " << sDFS/pDFS << endl;
	
	return 0;                                            
}