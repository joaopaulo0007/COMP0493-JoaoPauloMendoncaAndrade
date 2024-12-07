#ifndef PROCESSAMENTO_STRINGS_H
#define PROCESSAMENTO_STRINGS_H

#include <string>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;
typedef struct {
    int vogais;
    int consoantes;
    string textoLower;
} Resultado;

class ProcessamentoStrings {
public:
    ProcessamentoStrings();
    
    string processarArquivoTexto(const string& nomeArquivo);
    vector<int> findAllSubstrings(const string& texto, const string& padrao);
    Resultado countVogaisConsoantesAndtoLowercase(const string& texto);
    vector<string> quebrarTextoEmPalavras(const string& texto);
    vector<pair<string, int>>  contarPalavrasQueMaisRepetem(const vector<string>& palavras);
    int numCararcteresAposAlinhaDe7Pontos(const string& texto);
    
private:
};

#endif 