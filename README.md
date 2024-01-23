# heart-disease-pred
Machine learning models proposed to predict the degree of a heart disease.

## PL

Projekt został wykonany w ramach projektu semestralnego specjalizacji "Inteligentne systemy przetwarzania danych". Jest on propozycją dwóch modeli wykonanych w ramach algorytmu probabilistycznego Bayesa. Celem modeli jest wykrycie oznak choroby serca a następnie określenie czy dany pacjent wykazuje tendencje chorobowe na podstawie danych z badania lekarskiego.

### Model 1:

Podejście w pełni skupione na dokładnej ocenie czy pacjent jest zdrowy czy chory - model binarny. Wykazuje wyższą średnią dokładność - 86%.

### Model 2:

Podejście skupione na klasyfikacji stadium choroby pacjenta. Wykazuje mniejszą średnią dokładność - 64%



[!TIP]
Po więcej szczegółowych informacji, zachęcam do zapoznania się z raportem.

## ENG

Project created without external library implementation. This program uses perceptrons with a discrete activation function (neurons) to create a single-layer neural network. It is then used to detect language by injecting data into the program beforehand and training each perceptron on the principle of "english:other". This repository contains a sample training folder.

To use the program, follow these steps:

1. Create two folders to store training and testing data.
2. Create subfolders within them for each language and name them accordingly.
3. Fill the language subfolders with any number of text files containing text in that language.
4. Provide the training folder path and testing folder path as arguments.
