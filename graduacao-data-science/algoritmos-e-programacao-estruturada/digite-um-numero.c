#include <stdio.h>
#include <locale.h>

int main() {
  setlocale(LC_ALL, "Portuguese");

  int valor;

  printf("Digite um número: ");
  scanf("%d", &valor);

  printf("\nO número é:%d", valor);

  return 0;
}