#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

int main() {

  setlocale(LC_ALL, "Portuguese");
    
  float salarioBruto, inss, impostoRenda, salarioLiquido;

  // Solicitar o valor do salário bruto
  printf("Digite o valor do salário bruto: R$ ");
  scanf("%f", &salarioBruto);

  // Calcular o INSS (10% sobre o salário bruto)
  inss = salarioBruto * 0.1;

  // Calcular o imposto de renda (de acordo com a tabela progressiva)
  if (salarioBruto <= 1903.98) {
    impostoRenda = 0;
  } else if (salarioBruto <= 2826.65) {
    impostoRenda = (salarioBruto - 1903.98) * 0.075;
  } else if (salarioBruto <= 3751.05) {
    impostoRenda = (salarioBruto - 2826.65) * 0.15 + 71.54;
  } else if (salarioBruto <= 4664.68) {
    impostoRenda = (salarioBruto - 3751.05) * 0.225 + 336.90;
  } else {
    impostoRenda = (salarioBruto - 4664.68) * 0.275 + 636.13;
  }

  // Calcular o salário líquido
  salarioLiquido = salarioBruto - inss - impostoRenda;

  // Exibir os resultados
  printf("\nSalário Bruto: R$ %.2f", salarioBruto);
  printf("\nINSS: R$ %.2f", inss);
  printf("\nImposto de Renda: R$ %.2f", impostoRenda);
  printf("\nSalário Líquido: R$ %.2f", salarioLiquido);

  return 0;
}
