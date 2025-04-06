#ifndef LIB_H
#define LIB_H

#include <string>
#include <vector>
#include <map>
#include <cmath>

class Lib {
public:
    // String processing functions
    std::string readUntilSevenDots(const std::string& filename);
    std::vector<int> findAllOccurrences(const std::string& T, const std::string& P);
    std::string toLowerCase(const std::string& T);
    std::vector<std::string> tokenize(const std::string& T);
    std::string findSmallestToken(const std::string& T);
    std::vector<std::string> findMostFrequentWords(const std::string& T);
    int countLastLineCharacters(const std::string& filename);
    
    // String Algorithms
    std::vector<int> kmp(const std::string& text, const std::string& pattern);
    std::vector<int> boyerMoore(const std::string& text, const std::string& pattern);
    std::vector<int> rabinKarp(const std::string& text, const std::string& pattern);

    // Character analysis
    struct CharacterAnalysis {
        int digits;
        int vowels;
        int consonants;
    };
    CharacterAnalysis analyzeCharacters(const std::string& T);

    // Sorting algorithms
    std::vector<int> countingSort(std::vector<int> l);
    void bubble(std::vector<int> &lista, int tam);
    std::vector<int> bucketSort(std::vector<int> v, int tam);
    std::vector<int> radixSort(std::vector<int> lista);

    // Geometric functions
    struct Ponto {
        double x;
        double y;
    };
    double distancia2Pontos(Ponto a, Ponto b);
    double distanciaPontoReta(Ponto A, Ponto B, Ponto P);
    double areaSecaoTransversal(std::vector<Ponto>& pontos);

    // Graph functions
    struct Vertice {
        int vertice;
        std::vector<int> arestas;
    };
    void dfs(int v, std::vector<Vertice> &lista, std::vector<bool> &visitado, std::vector<Vertice> &verticesBusca);
    
    // Graph Flow Algorithms
    int fordFulkerson(std::vector<std::vector<int>>& graph, int source, int sink);
    int edmondsKarp(std::vector<std::vector<int>>& graph, int source, int sink);
    int dinic(std::vector<std::vector<int>>& graph, int source, int sink);
    
    // Union-Find and MST Algorithms
    struct DisjointSet {
        std::vector<int> parent, rank;
        DisjointSet(int n);
        int find(int x);
        void unionSets(int x, int y);
    };
    std::vector<std::pair<int, std::pair<int, int>>> kruskal(std::vector<std::pair<int, std::pair<int, int>>>& edges, int vertices);
    std::vector<std::pair<int, int>> prim(std::vector<std::vector<std::pair<int, int>>>& graph, int start);

    // Math functions
    long long binaryExponecial(int a, int b);
    int findMaior(std::vector<int> l);
    
    // Number Theory
    bool isPrimeNaive(int n);
    bool isPrimeOptimized(int n);
    long long modInverse(long long a, long long m);
    long long binomialCoefficient(int n, int k);
    long long binomialCoefficientDP(int n, int k);

    // Greedy Algorithms
    struct Item {
        double weight;
        double value;
    };
    double fractionalKnapsack(std::vector<Item>& items, double capacity);
    std::vector<int> coinChange(int amount, std::vector<int>& coins);
    std::vector<std::pair<int, int>> taskScheduling(std::vector<std::pair<int, int>>& tasks);

    // Divide and Conquer
    std::vector<int> mergeSort(std::vector<int>& arr);
    long long inversionCount(std::vector<int>& arr);
    std::string longestCommonPrefix(std::vector<std::string>& strs);

    // Graph Algorithms
    void bfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result);
    void dfsMatrix(std::vector<std::vector<int>>& graph, int start, std::vector<int>& result);
    void bfsList(std::vector<Vertice>& graph, int start, std::vector<int>& result);

    // Dynamic Programming Algorithms
    std::string longestCommonSubsequence(const std::string& text1, const std::string& text2);
    int knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity);
    int matrixChainMultiplication(const std::vector<int>& dimensions);
    int longestIncreasingSubsequence(const std::vector<int>& nums);
    
    // Data Structures
    struct FenwickTree {
        std::vector<int> bit;
        int size;
        
        FenwickTree(int n);
        FenwickTree(const std::vector<int>& arr);
        void update(int idx, int val);
        int query(int idx);
        int rangeQuery(int l, int r);
    };
    
    struct SegmentTree {
        std::vector<int> tree;
        std::vector<int> lazy;
        int size;
        
        SegmentTree(int n);
        SegmentTree(const std::vector<int>& arr);
        void build(const std::vector<int>& arr, int node, int start, int end);
        void updatePoint(int node, int start, int end, int idx, int val);
        void updateRange(int node, int start, int end, int l, int r, int val);
        int querySum(int node, int start, int end, int l, int r);
        int queryMin(int node, int start, int end, int l, int r);
        int queryMax(int node, int start, int end, int l, int r);
        void propagate(int node, int start, int end);
    };

private:
    bool isVowel(char c);
    bool isConsonant(char c);

    // Helper functions for divide and conquer
    void mergeSortHelper(std::vector<int>& arr, int left, int right);
    void merge(std::vector<int>& arr, int left, int mid, int right);
    long long mergeAndCount(std::vector<int>& arr, int left, int mid, int right);
    
    // Helper functions for string algorithms
    std::vector<int> computeLPS(const std::string& pattern);
    std::vector<int> buildBadCharTable(const std::string& pattern);
    std::vector<int> buildGoodSuffixTable(const std::string& pattern);
    
    // Helper functions for Ford-Fulkerson and variations
    bool bfs(std::vector<std::vector<int>>& residualGraph, int s, int t, std::vector<int>& parent);
    bool dinicBfs(const std::vector<std::vector<int>>& residualGraph, std::vector<int>& level, int s, int t);
    int dinicDfs(std::vector<std::vector<int>>& residualGraph, std::vector<int>& level, std::vector<int>& ptr, int u, int t, int flow);
};

#endif 