# Approche

Nous avons adopté l'algorithme basique du KNN, tel que présenté au TD3, et complémenté à l'aide de recherches en ligne ([towardsdatascience.com](https://towardsdatascience.com/) s'est prouvé très utile).

L'algorithme est ainsi très simple, puisque nous calculons simplement les distances entre l'individu testé et le dataset source. Intervient ensuite **k** , que nous avons auparavant testé sur les datasets fournis pour comparer les précisions.

# Tests

Nous nous sommes tout d'abord intéressé au paramètre **k** en prenant en compte différents calculs de distances provenant de **scipy.spatial.distances**. Nous avons déterminé que la distance euclidienne restait très bonne en précision, tout en étant la plus rapide à utiliser par rapport à d'autres méthodes.
Parmis les autres distances intéressantes figuraient celles de : 
- Chebyshev
- Manhattan
- Minkowski

Nous avons alors obtenu une précision de 0.732 pour un test entre le dataset d'origine et **preTest.txt**.

# Idées d'amélioration

Afin d'améliorer la précision de notre KNN, nous avons fait des recherches et déterminé qu'un preprocessing sur le dataset permettrait d'éliminer les "mauvais représentants" de chaque classe pour ne garder que les meilleurs, et ainsi améliorer la précision de l'algorithme final.
Nous n'avons toutefois pas eu le temps de mettre en place l'algorithme d'undersampling ([Tomek links](https://towardsdatascience.com/imbalanced-classification-in-python-smote-tomek-links-method-6e48dfe69bbc)) .

# Comparaison avec le KNN de scikit

Le fichier **sklearnKNN** contient l'implémentation du KNN en utilisant la bibliothèque scikit. On obtient une précision maximale de 74%. Nous ne sommes donc pas très loin de la précision que nous avons obtenue qui est de 73,2%.

Ci-dessous on peut voir la précision du modèle scikit selon le dataset utilisé et **k**

![alt text](http://url/to/img.png)
