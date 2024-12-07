#include "processamentoStrings.h"
#include <fstream>
#include <string>
#include <map>
#include <algorithm>
#include <sstream>
using namespace std;
ProcessamentoStrings::ProcessamentoStrings() {
}

string ProcessamentoStrings::processarArquivoTexto(const string& nomeArquivo) {
    ifstream arquivo(nomeArquivo);
    string textoFinal = "";
    string linha;
    int numLinhas = 0;
    
    while (getline(arquivo, linha) && numLinhas < 10) {
        if (linha.substr(0, 7) == ".......") {
            break;
        }
        
        if (!textoFinal.empty()) {
            textoFinal += " ";
        }
        
        textoFinal += linha;
        numLinhas++;
    }
    
    arquivo.close();
    return textoFinal;
} 

vector<int> ProcessamentoStrings::findAllSubstrings(const string& texto, const string& padrao) {
    vector<int> indices;
    size_t pos = texto.find(padrao);
    while (pos != string::npos) {
        indices.push_back(pos);
        pos = texto.find(padrao, pos + 1);
    }
    return indices;
}

Resultado ProcessamentoStrings::countVogaisConsoantesAndtoLowercase(const string& texto) {
    int vogais = 0, consoantes = 0;
    string textoLower = "";
    for (char c : texto) {
        if (isalpha(c)) {
            textoLower += tolower(c);
        }
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            vogais++;
        } else if (isalpha(c)) {
            consoantes++;
        }
    }
    return {vogais, consoantes, textoLower};
}
vector<string> ProcessamentoStrings::quebrarTextoEmPalavras(const string& texto) {
    vector<string> palavras;
    string palavraAtual;
    
    for (char c : texto) {
        if (c == ' ' || c == '.') {
            if (!palavraAtual.empty()) {
                palavras.push_back(palavraAtual);
                palavraAtual.clear();
            }
        } else {
            palavraAtual += c;
        }
    }
    if (!palavraAtual.empty()) {
        palavras.push_back(palavraAtual);
    }
    
    return palavras;
}

vector<pair<string, int>>  ProcessamentoStrings::contarPalavrasQueMaisRepetem(const vector<string>& palavras) {
    map<string, int> frequenciaPalavras;
    for (const string& palavra : palavras) {
        frequenciaPalavras[palavra]++;
    }
    vector<pair<string, int>> frequencias(frequenciaPalavras.begin(), frequenciaPalavras.end());
    sort(frequencias.begin(), frequencias.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
        return a.second > b.second;
    });
    return frequencias;
}

int ProcessamentoStrings::numCararcteresAposAlinhaDe7Pontos(const string& texto) {
    istringstream stream(texto);
    string linha;
    bool encontrouLinha = false;
    string textoRestante;
    
    while (getline(stream, linha)) {
        if (linha.substr(0, 7) == ".......") {
            encontrouLinha = true;
            continue;
        }
        if (encontrouLinha) {
            textoRestante += linha + "\n";
        }
    }
    
    return textoRestante.length();
}

