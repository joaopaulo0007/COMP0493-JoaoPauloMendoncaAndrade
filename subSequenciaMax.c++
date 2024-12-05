#include <iostream>
#include <vector>
#include <algorithm> 

int kadane(const std::vector<int> &vetor, int n) {
    int max_atual = vetor[0];
    int max_total = vetor[0];
    for (int i = 1; i < n; i++) {
        max_atual = std::max(vetor[i], max_atual + vetor[i]);
        max_total = std::max(max_total, max_atual);
    }
    return max_total;
}

int main() {
    int n;
    std::cin >> n; 
    std::vector<int> vetor(n);
    for (int i = 0; i < n; i++) {
        std::cin >> vetor[i];
    }
    std::cout << kadane(vetor, n) << "\n"; 
    return 0;
}
