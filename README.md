# Méthodes de Monte-Carlo

Cours de deuxième année à l'ENSAE Paris. 

## Objectifs
Idéalement, une méthode Monte-Carlo repose sur la simulation d’une suite de variables aléatoires
$(Xn)n≥1$ indépendantes et identiquement distribuées (i.i.d.) selon une loi donnée. La première partie 
du document Variables_aléatoires.ipynb expose quelques méthodes pour y parvenir, au moins de façon approchée, 
en commençant par la loi uniforme, sur laquelle toutes les autres sont basées. 

Plus généralement, nous abordons : 
- Les méthodes pour évaluer un algorithme (Loi forte des grands nombres, Test de Kolmogorov Smirnov, Algorithme de Floyd)
- Divers générateurs de nombres pseudos-aléatoires (Von Neuman, Fibonacci, générateurs congruentiels linéaires, Mersenne Twister)
- Nous simulons des variables aléatoires à l'aide des méthodes : d'inversion, de rejet, de changement de variable, ou via le TCL. 
