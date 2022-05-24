import sys

#code permettant de tester si un fichier de prédictions est au bon format.
#il prend en paramètre un fichier de labels prédits
#exemple d'utilisation > python checkLabels.py monFichierDePredictions.txt

allLabels = ['0','1']
#ce fichier s'attend à lire 1000 prédictions, une par ligne
#réduisez nbLines en période de test.
nbLines = 1000
fd =open(sys.argv[1],'r')
lines = fd.readlines()


count=0
for label in lines:
	if label.strip() in allLabels:
		count+=1
	else:
		break
if count==nbLines:
	print("Labels Check : Successfull!")
else:
	print("Wrong label line:"+str(count))
	print("Labels Check : fail! ", nbLines, "predictions expected" ,count, "found")

	


