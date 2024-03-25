#include <stdio.h>
#include <locale.h>

int main() {

  setlocale(LC_ALL, "Portuguese");

  // Declaração de variáveis
  float nota1, nota2, media;
  
  // Solicitação nota1
  printf("Insira nota 1:");
  scanf("%f", &nota1);

  // Solicitação nota2
  printf("Insira nota 2:");
  scanf("%f", &nota2);

  // Cálculo - Média
  media = (nota1 + nota2) / 2;

  // Apresentação das informações
  printf("\n**Cálculo: Média:**\n");
  printf("Nota 1: %f\n", nota1);
  printf("Nota 2: %f\n", nota2);
  printf("Média : %f\n", media);
  
  return 0;
}