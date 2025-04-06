#include "lib.h"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iostream>
#include <cmath>
#include <queue>
#include <climits>
#include <numeric>
#include <stack>

// String processing functions
std::string Lib::readUntilSevenDots(const std::string& filename) {
    std::ifstream file(filename);
    std::string result, line;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 7) == ".......") break;
        if (!result.empty()) result += " ";
        result += line;
    }
    
    return result;
}

std::vector<int> Lib::findAllOccurrences(const std::string& T, const std::string& P) {
    std::vector<int> positions;
    size_t pos = T.find(P);
    
    while (pos != std::string::npos) {
        positions.push_back(pos);
        pos = T.find(P, pos + 1);
    }
    
    return positions.empty() ? std::vector<int>{-1} : positions;
}

Lib::CharacterAnalysis Lib::analyzeCharacters(const std::string& T) {
    CharacterAnalysis analysis = {0, 0, 0};
    
    for (char c : T) {
        if (std::isdigit(c)) analysis.digits++;
        else if (isVowel(c)) analysis.vowels++;
        else if (isConsonant(c)) analysis.consonants++;
    }
    
    return analysis;
}

std::string Lib::toLowerCase(const std::string& T) {
    std::string result = T;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> Lib::tokenize(const std::string& T) {
    std::vector<std::string> tokens;
    std::stringstream ss(toLowerCase(T));
    std::string token;
    
    while (ss >> token) {
        token.erase(std::remove(token.begin(), token.end(), '.'), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    std::sort(tokens.begin(), tokens.end());
    return tokens;
}

std::string Lib::findSmallestToken(const std::string& T) {
    auto tokens = tokenize(T);
    return tokens.empty() ? "" : tokens[0];
}

std::vector<std::string> Lib::findMostFrequentWords(const std::string& T) {
    auto tokens = tokenize(T);
    std::map<std::string, int> frequency;
    int maxFreq = 0;
    
    for (const auto& token : tokens) {
        maxFreq = std::max(maxFreq, ++frequency[token]);
    }
    
    std::vector<std::string> mostFrequent;
    for (const auto& pair : frequency) {
        if (pair.second == maxFreq) {
            mostFrequent.push_back(pair.first);
        }
    }
    
    return mostFrequent;
}

int Lib::countLastLineCharacters(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::string lastLine;
    
    bool foundSeven = false;
    while (std::getline(file, line)) {
        if (foundSeven) {
            lastLine = line;
        }
        if (line.substr(0, 7) == ".......") {
            foundSeven = true;
        }
    }
    
    return lastLine.length();
}

bool Lib::isVowel(char c) {
    c = std::tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

bool Lib::isConsonant(char c) {
    return std::isalpha(c) && !isVowel(c);
}

// Sorting algorithms
int Lib::findMaior(std::vector<int> l) {
    int maior = l[0];
    for (int i = 1; i < l.size(); i++) {
        if (maior < l[i]) {
            maior = l[i];
        }
    }
    return maior;
}

std::vector<int> Lib::countingSort(std::vector<int> l) {
    if (l.empty())
        return l;
    int k = findMaior(l);
    std::vector<int> lista(k + 1, 0);
    for (int i = 0; i < l.size(); i++) {
        lista[l[i]] += 1;
    }
    for (int i = 1; i < k + 1; i++) {
        lista[i] += lista[i - 1];
    }
    std::vector<int> saida(l.size());
    for (int i = l.size() - 1; i >= 0; i--) {
        int valor = l[i];
        int posicaoSaida = lista[valor] - 1;
        saida[posicaoSaida] = valor;
        lista[valor]--;
    }
    return saida;
}

void Lib::bubble(std::vector<int> &lista, int tam) {
    int temp, flag;
    if (tam) {
        for (int i = 0; i < tam - 1; i++) {
            flag = 0;
            for (int j = 0; j < tam - 1; j++) {
                if (lista[j + 1] < lista[j]) {
                    temp = lista[j];
                    lista[j] = lista[j + 1];
                    lista[j + 1] = temp;
                    flag = 1;
                }
            }
            if (!flag) {
                break;
            }
        }
    }
}

std::vector<int> Lib::bucketSort(std::vector<int> v, int tam) {
    struct Bucket {
        int topo;
        std::vector<int> balde;
    };
    
    Bucket b[10];
    int i, j, k;
    for (i = 0; i < 10; i++)
        b[i].topo = 0;

    for (i = 0; i < tam; i++) {
        j = 9;
        while (1) {
            if (j < 0)
                break;
            if (v[i] >= j * 10) {
                b[j].balde.push_back(v[i]);
                (b[j].topo)++;
                break;
            }
            j--;
        }
    }

    for (i = 0; i < 10; i++)
        if (b[i].topo)
            bubble(b[i].balde, b[i].topo);

    i = 0;
    for (j = 0; j < 10; j++) {
        for (k = 0; k < b[j].topo; k++) {
            v[i] = b[j].balde[k];
            i++;
        }
    }
    return v;
}

std::vector<int> Lib::radixSort(std::vector<int> lista) {
    int maior = lista[0], exp = 1, tamanho = lista.size();
    std::vector<int> auxiliar(tamanho);
    for (int i = 1; i < tamanho; i++) {
        if (lista[i] > maior) {
            maior = lista[i];
        }
    }
    while (maior / exp > 0) {
        std::vector<int> baldes(10, 0);
        for (int i = 0; i < tamanho; i++) {
            baldes[(lista[i] / exp) % 10]++;
        }
        for (int i = 1; i < 10; i++)
            baldes[i] += baldes[i - 1];
        for (int i = tamanho - 1; i >= 0; i--)
            auxiliar[--baldes[(lista[i] / exp) % 10]] = lista[i];
        for (int i = 0; i < tamanho; i++)
            lista[i] = auxiliar[i];
        exp *= 10;
    }
    return lista;
}

// Geometric functions
double Lib::distancia2Pontos(Ponto a, Ponto b) {
    return hypot(a.x - b.x, a.y - b.y);
}

double Lib::distanciaPontoReta(Ponto A, Ponto B, Ponto P) {
    double numerador = fabs((B.x - A.x) * (A.y - P.y) - (A.x - P.x) * (B.y - A.y));
    double denominador = sqrt(pow(B.x - A.x, 2) + pow(B.y - A.y, 2));
    return numerador / denominador;
}

double Lib::areaSecaoTransversal(std::vector<Ponto>& pontos) {
    if (pontos.size() < 3) return 0.0; // Precisa de pelo menos 3 pontos para formar uma área
    
    double area = 0.0;
    int n = pontos.size();
    
    // Fórmula do Shoelace (Teorema de Green)
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += (pontos[i].x * pontos[j].y) - (pontos[j].x * pontos[i].y);
    }
    
    return fabs(area) / 2.0;
}

// Math functions
long long Lib::binaryExponecial(int a, int b) {
    if (b == 0) {
        return 1;
    }
    
    if (b % 2 == 0) {
        long long valor = binaryExponecial(a, b/2);
        return valor * valor;
    } else {
        long long valor = binaryExponecial(a, b/2);
        return a * valor * valor;
    }
}

// Graph functions
void Lib::dfs(int v, std::vector<Vertice> &lista, std::vector<bool> &visitado, std::vector<Vertice> &verticesBusca) {
    visitado[v] = true;
    verticesBusca.push_back(lista[v]);

    for (int vizinho : lista[v].arestas) {
        if (!visitado[vizinho]) {
            dfs(vizinho, lista, visitado, verticesBusca);
        }
    }
}

// Greedy Algorithms
double Lib::fractionalKnapsack(std::vector<Item>& items, double capacity) {
    // Sort items by value/weight ratio
    std::sort(items.begin(), items.end(), 
        [](const Item& a, const Item& b) {
            return (a.value / a.weight) > (b.value / b.weight);
        });
    
    double totalValue = 0.0;
    double currentWeight = 0.0;
    
    for (const Item& item : items) {
        if (currentWeight + item.weight <= capacity) {
            currentWeight += item.weight;
            totalValue += item.value;
        } else {
            double remainingWeight = capacity - currentWeight;
            totalValue += item.value * (remainingWeight / item.weight);
            break;
        }
    }
    
    return totalValue;
}

std::vector<int> Lib::coinChange(int amount, std::vector<int>& coins) {
    std::sort(coins.rbegin(), coins.rend()); // Sort in descending order
    std::vector<int> result;
    
    for (int coin : coins) {
        while (amount >= coin) {
            result.push_back(coin);
            amount -= coin;
        }
    }
    
    return result;
}

std::vector<std::pair<int, int>> Lib::taskScheduling(std::vector<std::pair<int, int>>& tasks) {
    // Sort by deadline
    std::sort(tasks.begin(), tasks.end(), 
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    std::vector<std::pair<int, int>> schedule;
    int currentTime = 0;
    
    for (const auto& task : tasks) {
        if (currentTime + task.first <= task.second) {
            schedule.push_back(task);
            currentTime += task.first;
        }
    }
    
    return schedule;
}

// Divide and Conquer
void Lib::mergeSortHelper(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void Lib::merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
}

std::vector<int> Lib::mergeSort(std::vector<int>& arr) {
    mergeSortHelper(arr, 0, arr.size() - 1);
    return arr;
}

long long Lib::mergeAndCount(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    long long inversions = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            inversions += mid - i + 1;
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }
    
    return inversions;
}

long long Lib::inversionCount(std::vector<int>& arr) {
    std::vector<int> temp = arr;
    return mergeAndCount(temp, 0, (temp.size() - 1) / 2, temp.size() - 1);
}

std::string Lib::longestCommonPrefix(std::vector<std::string>& strs) {
    if (strs.empty()) return "";
    if (strs.size() == 1) return strs[0];
    
    auto minmax = std::minmax_element(strs.begin(), strs.end());
    const std::string& first = *minmax.first;
    const std::string& last = *minmax.second;
    
    int i = 0;
    while (i < first.length() && i < last.length() && first[i] == last[i]) {
        i++;
    }
    
    return first.substr(0, i);
}

// Graph Algorithms
void Lib::bfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        result.push_back(v);
        
        for (int i = 0; i < n; i++) {
            if (graph[v][i] && !visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
    }
}

void Lib::dfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::stack<int> s;
    
    s.push(start);
    
    while (!s.empty()) {
        int v = s.top();
        s.pop();
        
        if (!visited[v]) {
            visited[v] = true;
            result.push_back(v);
            
            for (int i = n - 1; i >= 0; i--) {
                if (graph[v][i] && !visited[i]) {
                    s.push(i);
                }
            }
        }
    }
}

void Lib::bfsList(std::vector<Vertice>& graph, int start, std::vector<int>& result) {
    std::vector<bool> visited(graph.size(), false);
    std::queue<int> q;
    
    visited[start] = true;
    q.push(start);
    
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        result.push_back(v);
        
        for (int u : graph[v].arestas) {
            if (!visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
}

// Dynamic Programming Algorithms
std::string Lib::longestCommonSubsequence(const std::string& text1, const std::string& text2) {
    int m = text1.length();
    int n = text2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    
    // Fill the dp table
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    // Reconstruct the LCS
    std::string lcs;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (text1[i-1] == text2[j-1]) {
            lcs = text1[i-1] + lcs;
            i--; j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            i--;
        } else {
            j--;
        }
    }
    
    return lcs;
}

int Lib::knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
    int n = weights.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                dp[i][w] = std::max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w]);
            } else {
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    return dp[n][capacity];
}

int Lib::matrixChainMultiplication(const std::vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    
    // Length of chain
    for (int len = 2; len <= n; len++) {
        // Starting index of the chain
        for (int i = 0; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            
            // Try all possible splits
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + 
                          dimensions[i] * dimensions[k+1] * dimensions[j+1];
                dp[i][j] = std::min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n-1];
}

int Lib::longestIncreasingSubsequence(const std::vector<int>& nums) {
    if (nums.empty()) return 0;
    
    std::vector<int> dp(nums.size(), 1);
    int maxLen = 1;
    
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
                maxLen = std::max(maxLen, dp[i]);
            }
        }
    }
    
    return maxLen;
}

// Graph Flow Algorithms

// Helper function for Ford-Fulkerson and Edmonds-Karp
bool Lib::bfs(std::vector<std::vector<int>>& residualGraph, int s, int t, std::vector<int>& parent) {
    int n = residualGraph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < n; v++) {
            if (!visited[v] && residualGraph[u][v] > 0) {
                if (v == t) {
                    parent[v] = u;
                    return true;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
    return false;
}

int Lib::fordFulkerson(std::vector<std::vector<int>>& graph, int source, int sink) {
    int n = graph.size();
    std::vector<std::vector<int>> residualGraph = graph;
    std::vector<int> parent(n);
    int maxFlow = 0;
    
    // Augment the flow while there is a path from source to sink
    while (bfs(residualGraph, source, sink, parent)) {
        // Find the maximum flow through the path found
        int pathFlow = INT_MAX;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            pathFlow = std::min(pathFlow, residualGraph[u][v]);
        }
        
        // Update residual capacities of the edges and reverse edges
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }
        
        maxFlow += pathFlow;
    }
    
    return maxFlow;
}

int Lib::edmondsKarp(std::vector<std::vector<int>>& graph, int source, int sink) {
    // Edmonds-Karp is Ford-Fulkerson with BFS for finding augmenting paths
    // The implementation is already using BFS in the helper function
    return fordFulkerson(graph, source, sink);
}

bool Lib::dinicBfs(const std::vector<std::vector<int>>& residualGraph, std::vector<int>& level, int s, int t) {
    int n = residualGraph.size();
    std::fill(level.begin(), level.end(), -1);
    level[s] = 0;
    
    std::queue<int> q;
    q.push(s);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v = 0; v < n; v++) {
            if (level[v] < 0 && residualGraph[u][v] > 0) {
                level[v] = level[u] + 1;
                q.push(v);
            }
        }
    }
    
    return level[t] >= 0;
}

int Lib::dinicDfs(std::vector<std::vector<int>>& residualGraph, std::vector<int>& level, 
                 std::vector<int>& ptr, int u, int t, int flow) {
    if (u == t)
        return flow;
    
    int n = residualGraph.size();
    for (int& i = ptr[u]; i < n; i++) {
        int v = i;
        if (level[v] == level[u] + 1 && residualGraph[u][v] > 0) {
            int curr_flow = std::min(flow, residualGraph[u][v]);
            int temp_flow = dinicDfs(residualGraph, level, ptr, v, t, curr_flow);
            
            if (temp_flow > 0) {
                residualGraph[u][v] -= temp_flow;
                residualGraph[v][u] += temp_flow;
                return temp_flow;
            }
        }
    }
    
    return 0;
}

int Lib::dinic(std::vector<std::vector<int>>& graph, int source, int sink) {
    int n = graph.size();
    std::vector<std::vector<int>> residualGraph = graph;
    std::vector<int> level(n);
    std::vector<int> ptr(n);
    int maxFlow = 0;
    
    while (dinicBfs(residualGraph, level, source, sink)) {
        std::fill(ptr.begin(), ptr.end(), 0);
        while (int flow = dinicDfs(residualGraph, level, ptr, source, sink, INT_MAX))
            maxFlow += flow;
    }
    
    return maxFlow;
}

// Union-Find Data Structure
Lib::DisjointSet::DisjointSet(int n) {
    parent.resize(n);
    rank.resize(n, 0);
    for (int i = 0; i < n; i++)
        parent[i] = i;
}

int Lib::DisjointSet::find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

void Lib::DisjointSet::unionSets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    
    if (rootX == rootY)
        return;
    
    if (rank[rootX] < rank[rootY])
        parent[rootX] = rootY;
    else if (rank[rootX] > rank[rootY])
        parent[rootY] = rootX;
    else {
        parent[rootY] = rootX;
        rank[rootX]++;
    }
}

// Kruskal's algorithm for MST
std::vector<std::pair<int, std::pair<int, int>>> Lib::kruskal(
    std::vector<std::pair<int, std::pair<int, int>>>& edges, int vertices) {
    
    std::sort(edges.begin(), edges.end()); // Sort edges by weight
    DisjointSet ds(vertices);
    std::vector<std::pair<int, std::pair<int, int>>> result;
    
    for (auto& edge : edges) {
        int weight = edge.first;
        int u = edge.second.first;
        int v = edge.second.second;
        
        if (ds.find(u) != ds.find(v)) {
            result.push_back(edge);
            ds.unionSets(u, v);
        }
    }
    
    return result;
}

// Prim's algorithm for MST
std::vector<std::pair<int, int>> Lib::prim(std::vector<std::vector<std::pair<int, int>>>& graph, int start) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::vector<int> key(n, INT_MAX);
    std::vector<int> parent(n, -1);
    
    // Use priority queue to find minimum weight edge
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, 
                       std::greater<std::pair<int, int>>> pq;
    
    // Start with vertex 'start'
    key[start] = 0;
    pq.push({0, start});
    
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        
        if (visited[u])
            continue;
        
        visited[u] = true;
        
        for (auto& neighbor : graph[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            
            if (!visited[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                pq.push({key[v], v});
            }
        }
    }
    
    // Construct the MST edges
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < n; i++) {
        if (i != start && parent[i] != -1) {
            result.push_back({parent[i], i});
        }
    }
    
    return result;
}

// Number Theory Functions
bool Lib::isPrimeNaive(int n) {
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    
    return true;
}

bool Lib::isPrimeOptimized(int n) {
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    
    // Check using 6k ± 1 optimization
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    
    return true;
}

// Modular multiplicative inverse using Extended Euclidean Algorithm
long long Lib::modInverse(long long a, long long m) {
    long long m0 = m;
    long long y = 0, x = 1;
    
    if (m == 1)
        return 0;
    
    while (a > 1) {
        long long q = a / m;
        long long t = m;
        
        m = a % m;
        a = t;
        t = y;
        
        y = x - q * y;
        x = t;
    }
    
    if (x < 0)
        x += m0;
    
    return x;
}

// Binomial Coefficient (analytical approach)
long long Lib::binomialCoefficient(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    
    // C(n, k) = C(n, n-k)
    if (k > n - k)
        k = n - k;
    
    long long res = 1;
    
    // Calculate [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    
    return res;
}

// Binomial Coefficient (DP approach)
long long Lib::binomialCoefficientDP(int n, int k) {
    std::vector<std::vector<long long>> C(n + 1, std::vector<long long>(k + 1, 0));
    
    // Base cases
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= std::min(i, k); j++) {
            if (j == 0 || j == i)
                C[i][j] = 1;
            else
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }
    
    return C[n][k];
}

// String Matching Algorithms
std::vector<int> Lib::computeLPS(const std::string& pattern) {
    int m = pattern.length();
    std::vector<int> lps(m, 0);
    
    int len = 0;
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}

std::vector<int> Lib::kmp(const std::string& text, const std::string& pattern) {
    int n = text.length();
    int m = pattern.length();
    std::vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    // Preprocess pattern to get longest prefix suffix array
    std::vector<int> lps = computeLPS(pattern);
    
    int i = 0; // Index for text[]
    int j = 0; // Index for pattern[]
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches.push_back(i - j); // Found a match
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
    
    return matches;
}

std::vector<int> Lib::buildBadCharTable(const std::string& pattern) {
    int m = pattern.length();
    std::vector<int> badChar(256, -1);
    
    for (int i = 0; i < m; i++)
        badChar[pattern[i]] = i;
    
    return badChar;
}

std::vector<int> Lib::buildGoodSuffixTable(const std::string& pattern) {
    int m = pattern.length();
    std::vector<int> shift(m, 0);
    std::vector<int> border(m, 0);
    
    // Preprocessing for case 2
    int j = m;
    border[m - 1] = j;
    
    for (int i = m - 2; i >= 0; i--) {
        while (j < m && pattern[i] != pattern[j - 1])
            j = border[j];
        
        j--;
        border[i] = j;
    }
    
    // Preprocessing for case 1
    for (int i = 0; i < m; i++)
        shift[i] = m;
    
    j = 0;
    for (int i = m - 1; i >= 0; i--) {
        if (border[i] == i + 1) {
            while (j < m - 1 - i)
                shift[j++] = m - 1 - i;
        }
    }
    
    for (int i = 0; i <= m - 2; i++)
        shift[m - 1 - border[i]] = m - 1 - i;
    
    return shift;
}

std::vector<int> Lib::boyerMoore(const std::string& text, const std::string& pattern) {
    int n = text.length();
    int m = pattern.length();
    std::vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    // Preprocess pattern
    std::vector<int> badChar = buildBadCharTable(pattern);
    std::vector<int> goodSuffix = buildGoodSuffixTable(pattern);
    
    int s = 0; // Shift of the pattern relative to text
    
    while (s <= n - m) {
        int j = m - 1;
        
        // Match pattern from right to left
        while (j >= 0 && pattern[j] == text[s + j])
            j--;
        
        if (j < 0) {
            matches.push_back(s); // Pattern found at position s
            s += (s + m < n) ? m - badChar[text[s + m]] : 1;
        } else {
            // Bad Character heuristic
            int badCharShift = j - badChar[text[s + j]];
            if (badCharShift < 1) badCharShift = 1;
            
            // Good Suffix heuristic
            int goodSuffixShift = goodSuffix[j];
            
            s += std::max(badCharShift, goodSuffixShift);
        }
    }
    
    return matches;
}

std::vector<int> Lib::rabinKarp(const std::string& text, const std::string& pattern) {
    int n = text.length();
    int m = pattern.length();
    std::vector<int> matches;
    
    if (m == 0 || m > n)
        return matches;
    
    const int prime = 101; // A prime number
    const int d = 256;     // Number of characters in the alphabet
    
    // Calculate hash for pattern and first window of text
    int patternHash = 0;
    int textHash = 0;
    int h = 1;
    
    // Calculate h = pow(d, m-1) % prime
    for (int i = 0; i < m - 1; i++)
        h = (h * d) % prime;
    
    // Calculate hash value for pattern and first window of text
    for (int i = 0; i < m; i++) {
        patternHash = (d * patternHash + pattern[i]) % prime;
        textHash = (d * textHash + text[i]) % prime;
    }
    
    // Slide the pattern over text one by one
    for (int i = 0; i <= n - m; i++) {
        // Check if the hash values match
        if (patternHash == textHash) {
            // Check characters one by one
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match)
                matches.push_back(i);
        }
        
        // Calculate hash value for next window of text
        if (i < n - m) {
            textHash = (d * (textHash - text[i] * h) + text[i + m]) % prime;
            
            // We might get negative hash, convert it to positive
            if (textHash < 0)
                textHash += prime;
        }
    }
    
    return matches;
}

// Fenwick Tree (Binary Indexed Tree) implementation
Lib::FenwickTree::FenwickTree(int n) {
    size = n;
    bit.assign(n + 1, 0);
}

Lib::FenwickTree::FenwickTree(const std::vector<int>& arr) {
    size = arr.size();
    bit.assign(size + 1, 0);
    
    for (int i = 0; i < size; i++)
        update(i, arr[i]);
}

void Lib::FenwickTree::update(int idx, int val) {
    idx++; // 1-based indexing
    while (idx <= size) {
        bit[idx] += val;
        idx += idx & -idx; // Add LSB
    }
}

int Lib::FenwickTree::query(int idx) {
    idx++; // 1-based indexing
    int sum = 0;
    while (idx > 0) {
        sum += bit[idx];
        idx -= idx & -idx; // Remove LSB
    }
    return sum;
}

int Lib::FenwickTree::rangeQuery(int l, int r) {
    return query(r) - query(l - 1);
}

// Segment Tree implementation
Lib::SegmentTree::SegmentTree(int n) {
    size = n;
    tree.assign(4 * n, 0);
    lazy.assign(4 * n, 0);
}

Lib::SegmentTree::SegmentTree(const std::vector<int>& arr) {
    size = arr.size();
    tree.assign(4 * size, 0);
    lazy.assign(4 * size, 0);
    build(arr, 1, 0, size - 1);
}

void Lib::SegmentTree::build(const std::vector<int>& arr, int node, int start, int end) {
    if (start == end) {
        tree[node] = arr[start];
        return;
    }
    
    int mid = (start + end) / 2;
    build(arr, 2 * node, start, mid);
    build(arr, 2 * node + 1, mid + 1, end);
    tree[node] = tree[2 * node] + tree[2 * node + 1]; // Sum query
}

void Lib::SegmentTree::propagate(int node, int start, int end) {
    if (lazy[node] != 0) {
        tree[node] += (end - start + 1) * lazy[node]; // Update node value
        
        if (start != end) {
            lazy[2 * node] += lazy[node];     // Mark child as lazy
            lazy[2 * node + 1] += lazy[node]; // Mark child as lazy
        }
        
        lazy[node] = 0; // Reset lazy value
    }
}

void Lib::SegmentTree::updatePoint(int node, int start, int end, int idx, int val) {
    if (start == end) {
        tree[node] = val;
        return;
    }
    
    int mid = (start + end) / 2;
    if (idx <= mid)
        updatePoint(2 * node, start, mid, idx, val);
    else
        updatePoint(2 * node + 1, mid + 1, end, idx, val);
    
    tree[node] = tree[2 * node] + tree[2 * node + 1]; // Sum query
}

void Lib::SegmentTree::updateRange(int node, int start, int end, int l, int r, int val) {
    propagate(node, start, end);
    
    if (start > end || start > r || end < l)
        return;
    
    if (start >= l && end <= r) {
        tree[node] += (end - start + 1) * val;
        
        if (start != end) {
            lazy[2 * node] += val;
            lazy[2 * node + 1] += val;
        }
        
        return;
    }
    
    int mid = (start + end) / 2;
    updateRange(2 * node, start, mid, l, r, val);
    updateRange(2 * node + 1, mid + 1, end, l, r, val);
    
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

int Lib::SegmentTree::querySum(int node, int start, int end, int l, int r) {
    if (start > end || start > r || end < l)
        return 0;
    
    propagate(node, start, end);
    
    if (start >= l && end <= r)
        return tree[node];
    
    int mid = (start + end) / 2;
    int p1 = querySum(2 * node, start, mid, l, r);
    int p2 = querySum(2 * node + 1, mid + 1, end, l, r);
    
    return p1 + p2;
}

int Lib::SegmentTree::queryMin(int node, int start, int end, int l, int r) {
    if (start > end || start > r || end < l)
        return INT_MAX;
    
    propagate(node, start, end);
    
    if (start >= l && end <= r)
        return tree[node];
    
    int mid = (start + end) / 2;
    int p1 = queryMin(2 * node, start, mid, l, r);
    int p2 = queryMin(2 * node + 1, mid + 1, end, l, r);
    
    return std::min(p1, p2);
}

int Lib::SegmentTree::queryMax(int node, int start, int end, int l, int r) {
    if (start > end || start > r || end < l)
        return INT_MIN;
    
    propagate(node, start, end);
    
    if (start >= l && end <= r)
        return tree[node];
    
    int mid = (start + end) / 2;
    int p1 = queryMax(2 * node, start, mid, l, r);
    int p2 = queryMax(2 * node + 1, mid + 1, end, l, r);
    
    return std::max(p1, p2);
} 