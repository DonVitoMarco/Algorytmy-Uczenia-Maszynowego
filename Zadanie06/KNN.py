import numpy as np


# ### K-Nearest Neighbors implementacja ###
# W celu klasyfikacji obliczana jest odległość czyli podobieństwo od obiektu który chcemy
# sklasyfikować do obiektów z zestawu testowego.
# Podobieństwo może być wyliczne w różny sposób w przypadku tej implementacji
# będzie to odległość euklidesowa.
class KNN:
    def __init__(self, k) -> None:
        self.k = k

    # Metoda statyczna wyliczająca odlekłość euklidesową pomiędzy dwoma wektorami.
    @staticmethod
    def euclidean(v1, v2):
        vector1 = np.array(v1)
        vector2 = np.array(v2)
        distance = 0
        for i in range(len(vector1) - 1):
            distance += (vector1[i] - vector2[i]) ** 2
        return np.sqrt(distance)

    # Prosta ocena naczego algorytmu,
    # weryfikacja w ilu przypadkach algorytm poprawnie zaklasyfikował obiekt.
    @staticmethod
    def evaluate(y, y_pred):
        correct = 0
        best = len(y)
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                correct += 1
        return correct / best

    def predict(self, train_set, test_object):
        # Oblicz odległość euklidesową między obiektem do klasyfikacji, 
        # a każdą obiektem testowym i zapisz ją do tablicy distances.
        distances = []
        for i in range(len(train_set)):
            distance = self.euclidean(train_set.iloc[i][:-1], test_object)
            distances.append((train_set.iloc[i], distance))
        # Posortuj uzyskane dystansy od najkrótszego do najdłuższego
        distances.sort(key=lambda x: x[1])

        # Z posortowanych dystansów weź k najkrótszych odległości.
        # Czyli k początkowych elementów tablicy.
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])


        classes = {}
        # Sprawdź jakiej klasy są k najblisi sąsiedzi.
        for i in range(len(neighbors)):
            # Pobierz klasę dla sąsiada.
            response = neighbors[i][-1]
            # Jeżeli wcześniej już był sąsiad takiej klasy zwiększa wartość dla tej klasy o 1.
            if response in classes:
                classes[response] += 1
            # Jeżeli jest to klasa która wcześniej nie występowała dodaj ją do obiektu classes.
            else:
                classes[response] = 1

        # Po posortowaniu wartości zapisanych w classes, wiemy która klasa występowała najczęściej wśród
        # sąsiadów i do takiej klasy klasyfikujemy nasz obiekt.
        return sorted(classes.items(), key=lambda x: x[1], reverse=True)[0][0]
