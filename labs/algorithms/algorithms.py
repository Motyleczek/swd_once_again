# Michał Motyl 401943
# Kamil Pieprzycki 402037
# Poniższy plik był wykonywany wspólnie podział kolejno w 8 i 89 linijce
import numpy as np
import time
from typing import Tuple

np.random.seed(123)

## Część Michała - wersja algorytmu naiwnego bez i z filtracją 

# funkcja do porównywania punktów: wariant z minimalizacją wszystkich parametrów
# tej funkcji używają wszystkie kolejne algorytmy i napisaliśmy ją wspólnie

def compare(y, x):
    # realizuje porównanie y <= x
    truth_tab = []
    n = len(y)
    for i in range(n):
        truth_tab.append(y[i] <= x[i])
    summary = sum(truth_tab)
    if summary == n:
        return True
    else:
        return False


# wersja naiwna bez filtracji:
def naiwne_owd_v1(x: list)->Tuple:
    """
    Algorytm naiwny wersja 1
    
    params:
    x - lista punktów do posortowania
    
    returns:
    P - punkty niezdominowane
    L - liczba porównań
    total_time - ile czasu zajęło porównanie
    """
    n = len(x)
    X = x[:]
    P = []
    # do zliczania liczby porównań:
    L = 0
    # do sprawdzania czasu obliczeń:
    start = time.time()

    for i in range(n):
        if len(X) == 0:
            break
        X_alter = X[:]
        m = len(X_alter)
        Y = X[0]
        fl = 0
        for j in range(1, m):
            if compare(Y, X[j]):
                X_alter.remove(X[j])
            elif compare(X[j], Y):
                X_alter.remove(Y)
                Y = X[j]
                fl = 1
            L += 1

        if Y not in P:
            P.append(Y)
        if fl == 0:
            X_alter.remove(Y)
        X = X_alter[:]
    total_time = time.time() - start

    return P, L, total_time
  


# wersja naiwna z filtracją:
def naiwne_owd_v2(x: list):
    """
    Algorytm naiwny wersja 2
    
    params:
    x - lista punktów do posortowania
    
    returns:
    P - punkty niezdominowane
    L - liczba porównań
    total_time - ile czasu zajęło porównanie
    """
    n = len(x)
    X = x[:]
    P = []
    # do zliczania liczby porównań:
    L = 0
    # do sprawdzania czasu obliczeń:
    start = time.time()

    for i in range(n):
        if len(X) == 0:
            break
        X_alter = X[:]
        m = len(X_alter)
        Y = X[0]
        for j in range(1, m):
            if compare(Y, X[j]):
                X_alter.remove(X[j])
            elif compare(X[j], Y):
                X_alter.remove(Y)
                Y = X[j]
            L += 1

        if Y not in P:
            P.append(Y)

        # poniżej bez kopii [:] listy iteracja po liście nam przeskoczy elementy po usunięciu,
        # dlatego musimy tej kopii użyć
        for elem in X_alter[:]:
            if compare(Y, elem):
                X_alter.remove(elem)
            L += 1

        if Y in X_alter:
            X_alter.remove(Y)
        X = X_alter[:]

    total_time = time.time() - start
    return P, L, total_time

## część Kamila - algorytm oparty o punkt idealny

def naiwne_owd_v3(x: list):
    """
    Sortowanie o punkt idealny
    
    params:
    x - lista punktów do posortowania
    
    returns:
    P - punkty niezdominowane
    L - liczba porównań
    total_time - ile czasu zajęło porównanie
    """
    X = np.array(x)
    X_list = x[:]
    X_altered = X_list[:]
    n = len(x)
    P = []
    # do zliczania porównań:
    L = 0
    # do sprawdzania czasu obliczeń:
    start = time.time()

    # znajdujemy współrzędne naszego punktu idealnego:
    # numpy to znacząco ułatwia, zakładamy tylko że podane dane na wejście
    # są w odpowiedniej formie (lista krotek)
    xmin = np.min(X, axis=0)

    # dummy d:
    d = np.array([np.nan, np.nan])

    for i in range(n):
        distance_sq = (np.linalg.norm(xmin-X[i, :]))**2
        if i == 0:
            d = np.array([i, distance_sq])
        else:
            d = np.vstack([d, [i, distance_sq]])
    d_sorted = d[np.argsort(d[:, 1])]

    M = n
    m = 0
    while m <= M:
        X_temporary_list = X_altered[:]
        X_jm = X_list[int(d_sorted[m, 0])]

        # poniżej skipujemy jeden punkt, jak już go usunęliśmy, bez zmniejszania M
        if X_jm not in X_altered:
            m += 1
            continue

        # usuń z X wszystkie X(i) takie, że X(J(m))≤X(i);
        for elem in X_altered:
            if compare(X_jm, elem):
                X_temporary_list.remove(elem)
            L += 1

        # usuń X(J(m)) z X;
        if X_jm in X_temporary_list:
            X_temporary_list.remove(X_jm)

        # dodaj X(J(m)) do listy punktów niezdominowanych P;
        P.append(X_jm)

        # aktualizacja zmiennych
        X_altered = X_temporary_list[:]
        M -= 1
        m += 1

    total_time = time.time() - start
    return P, L, total_time


