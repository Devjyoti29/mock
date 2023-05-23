#include <iostream>
#include <omp.h>
#include <vector>
#include <queue>
#include <iomanip>
#include <time.h>
#include <stack>

using namespace std;

class Graph
{
public:
    int V;
    vector<vector<int>> graph;
    Graph(int vertices)
    {
        V = vertices;
        graph.resize(vertices);
    }

    void addEdge(int u, int v)
    {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    void bfs(int src)
    {
        int *visited = new int[V];
        queue<int> q;
        q.push(src);
        visited[src] = 1;

        while (!q.empty())
        {
            int node = q.front();
            q.pop();

            // cout << node << " ";

            for (int ele : graph[node])
            {
                if (visited[ele] != 1)
                {
                    visited[ele] = 1;
                    q.push(ele);
                }
            }
        }
    }
    void dfs(int src)
    {
        int *visited = new int[V];
        queue<int> q;
        q.push(src);
        visited[src] = 1;

        while (!q.empty())
        {
            int node = q.front();
            q.pop();

            // cout << node << " ";

            for (int ele : graph[node])
            {
                if (visited[ele] != 1)
                {
                    visited[ele] = 1;
                    q.push(ele);
                }
            }
        }
    }

    void parallel_bfs(int src)
    {
        int *visited = new int[V];
        queue<int> q;
        q.push(src);
        visited[src] = 1;

        while (!q.empty())
        {
            int node = q.front();
            q.pop();

            // cout << node << " ";

#pragma omp parallel for
            for (int i = 0; i < graph[node].size(); i++)
            {
                int ele = graph[node][i];
                int visited_ele = 0;

#pragma omp atomic read
                visited_ele = visited[ele];

                if (visited_ele != 1)
                {
#pragma omp atomic write
                    visited[ele] = 1;

#pragma omp critical
                    q.push(ele);
                }
            }
        }
    }
    void parallel_dfs(int src)
    {
        int *visited = new int[V];
        stack<int> q;
        q.push(src);
        visited[src] = 1;

        while (!q.empty())
        {
            int node = q.top();
            q.pop();

            // cout << node << " ";

#pragma omp parallel for
            for (int i = 0; i < graph[node].size(); i++)
            {
                int ele = graph[node][i];
                int visited_ele = 0;

#pragma omp atomic read
                visited_ele = visited[ele];

                if (visited_ele != 1)
                {
#pragma omp atomic write
                    visited[ele] = 1;

#pragma omp critical
                    q.push(ele);
                }
            }
        }
    }
};

int main()
{
    int N;
    cout << "Number of Nodes : ";
    cin >> N;

    Graph g(N);

    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            if ((i * 3 + j) % 7 == 0 || (i + 2 * j) % 5 == 0)
            {
                g.addEdge(i, j);
            }
        }
    }

    clock_t start, end;

    start = clock();
    g.bfs(0);
    end = clock();

    float CPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "CPU Time  for BFS  : " << CPUTime << endl;

    start = clock();
    g.parallel_bfs(0);
    end = clock();

    float GPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "GPU Time  for BFS : " << GPUTime << endl;

    cout << "SpeedUp  for BFS  : " << fixed << setprecision(9) << CPUTime / GPUTime << endl;

    start = clock();
    g.dfs(0);
    end = clock();

    CPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "CPU Time for DFS : " << CPUTime << endl;

    start = clock();
    g.parallel_dfs(0);
    end = clock();

    GPUTime = ((float)(end - start)) / CLOCKS_PER_SEC;
    cout << "GPU Time for DFS : " << GPUTime << endl;

    cout << "SpeedUp for DFS  : " << fixed << setprecision(9) << CPUTime / GPUTime << endl;
}