#include <stdio.h>
#include <locale.h>

int main() {

  setlocale(LC_ALL, "Portuguese");

  // Declaração de variáveis
  char nome[50];
  char endereco[100];
  char cidade[50];
  int idade;

  // Solicitação do nome
  printf("Digite seu nome: ");
  scanf(" %[^\n]%*c", nome); // Leitura com espaço em branco

  // Solicitação do endereço
  printf("Digite seu endereço: ");
  scanf(" %[^\n]%*c", endereco);

  // Solicitação da cidade
  printf("Digite sua cidade: ");
  scanf(" %[^\n]%*c", cidade);

  // Solicitação da idade
  printf("Digite sua idade: ");
  scanf("%d", &idade);

  // Apresentação das informações
  printf("\n**Dados Cadastrados:**\n");
  printf("Nome: %s\n", nome);
  printf("Endereço: %s\n", endereco);
  printf("Cidade: %s\n", cidade);
  printf("Idade: %d anos\n", idade);

  return 0;
}