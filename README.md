# Közlekedési tábla felismerő program

A programnak beadunk egy mp4 formátumú fájlt, majd az készít egy predikciót rá fél másodpercenként, hogy milyen táblák láthatók a képen.

Jupyter Notebookot használtam, és Python 3.9.21-es verziót (viszont a 3.9.13-at töltöttem le: https://www.python.org/downloads/release/python-3913/), amelyre létrehoztam egy külön "tf_env" (tensorflow envorinment) környezetet. 

A requirements.txt-ben lévő verzióknak meg kell felelniük, különösen fontos a TensorFlow verzió, és a numpy verzió.

Használat:
1. Le kell futtatni a "Save Images" fejezetig mindent,
2. ezután a "Load Model" fejezetben be kell tölteni a modellt ami már meg is van adva,
3. majd a "Working Program" fejezetben először készíteni kell a videó nevével egy csv fájlt az első, "Prediction for videos to CSV files" nevű blokkal, majd a második, "Visual representation for the predictions" blokkal be kell tölteni ezt a videót és csv fájlt, és utána futtatni kell, hogy megnyíljon a vizuális felhasználói interface amin látjuk a képeket és rájuk a predikciókat.

