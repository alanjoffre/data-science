#include <stdio.h>
#include <stdlib.h>
#include <locale.h>

int main() {

  setlocale(LC_ALL, "Portuguese");
    
    int i; 
    do{
        printf ("\n INFORMACAO NUTRICIONAL DO SORVETE\n");
        printf ("\n Digite um número que corresponde ao saber desejado\n");
        
        printf ("\t(1)...flocos\n");
        printf ("\t(2)...morango\n");
        printf ("\t(3)...leite condensado\n\n"); 
        scanf("%d", &i);
        
    } while ((i<1)||(i>3));
    
    switch (i) {
        case 1:
        printf ("\t\tVoce escolheu flocos.\n");
        break; 
        
        case 2:
        printf ("\t\tVoce escolheu morango.\n"); 
        break;
        
        case 3: 
        printf ("\t\tVoce escolheu leite condensado.\n"); break;
        
    }
    return(0);
}
