import numpy as np
import time
from matplotlib import pyplot as matpl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

start_time = time.time()  # pentru a masura eficienta de timp a programului

# Citirea si prelucrarea fisierului train.txt
train_aux = np.loadtxt('./train.txt', dtype='str')[1:]
train_aux = [x.split(',') for x in train_aux]
train_images_names = [x[0] for x in train_aux]
train_images_labels = [int(x[1]) for x in train_aux]

# Citirea fisierului test.txt
test_images_names = np.loadtxt('./test.txt', dtype='str')[1:]

# Citirea si prelucrarea fisierului validation.txt
validation_aux = np.loadtxt('./validation.txt', dtype='str')[1:]
validation_aux = [x.split(',') for x in validation_aux]
validation_images_names = [x[0] for x in validation_aux]
validation_images_labels = [int(x[1]) for x in validation_aux]

# citim imaginile cu imread din libraria matplotlib
train_images = np.array([matpl.imread(f"train+validation/{name}") for name in train_images_names])

# Transformam(flatten) np array-ul dintr-unul (8000, 16, 16, 3) intr-unul (8000, 16*16*3) = (8000, 768), unde pentru
# fiecare linie(imagine) primele 256 sunt valorile RED , urmatoarele 256 valorile GREEN iar ultimele 256 valorile BLUE
train_images = np.array(
    [np.concatenate([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()]) for img in train_images])

# Repetam procesul de la train_images si pentru validation si test images
validation_images = np.array([matpl.imread(f"train+validation/{name}") for name in validation_images_names])
validation_images = np.array(
    [np.concatenate([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()]) for img in
     validation_images])

test_images = np.array([matpl.imread(f"test/{name}") for name in test_images_names])
test_images = np.array(
    [np.concatenate([img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()]) for img in test_images])


# In caz ca vrem sa afisam o imagine:
# matpl.imshow(train_images[0])
# matpl.show()


# Primul clasificator: KNN, K nearest neighbours, care face o predictie bazat pe
# cei mai apropiati vecini din tot setul de imagini de antrenare
class KNN:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    # Functia de clasificare
    def classifyItem(self, image, neighbours_count=5):
        # Calculam distantele dintre imaginea data spre clasificare si toate imaginile din train_images.
        # Folosim distana L2
        allDist = np.sqrt(np.sum((self.train_images - image) ** 2, axis=1))

        # Luam cu argsort pozitiile celor mai mici 'neighbouts_count'(5 in cazul default) distante
        sortedImagesIndexes = np.argsort(allDist)[:neighbours_count]

        # Scoatem labels-urile imaginilor de pe pozitiile calculate in sortedImagesIndexes
        # (deci cele mai apropiate imagini)
        neighbours = np.array(self.train_labels)[sortedImagesIndexes]

        # Generam un vector de frecventa cu labels-urile din neighbours, ca sa vedem care este
        # cea mai posibila eticheta pt imaginea data spre clasificare
        frequencyForLabels = np.bincount(neighbours)

        # Intoarcem pozitia cu valoarea maxima in vectorul de frecventa ca fiind predictia
        return np.argmax(frequencyForLabels)

    # Metoda ce aplica KNN classification pe un set de imagini
    def applyClassification(self, images, neighbours_count=5):
        # Toate predictiile sunt initial 0
        predictions = np.zeros((images.shape[0]), int)

        for poz in range(images.shape[0]):
            # Clasificam a 'poz' - a imagine
            predictions[poz] = self.classifyItem(images[poz, :], neighbours_count)

        return predictions

    # Functie ce compara predictiile clasificatorului cu etichetele adevarate
    # si intoarce acuratetea drept procent(subunitar)
    def getAccuracy(self, predicts, reals):
        aux = np.array([predicts[i] == reals[i] for i in range(len(predicts))])
        return aux.mean()


# Cream o instanta a clasei KNN si ii pasam datele de antrenament
knn = KNN(train_images, train_images_labels)

# Testam acuratetea modelului pe datele de validare
# Dupa multiple incercari am observat ca pentru 5 vecini functioneaza optim, deci vom rula cu num_neighbours = 5
predictions = knn.applyClassification(validation_images, 5)
accuracy = knn.getAccuracy(predictions, validation_images_labels)
print('KNN accuracy with 5 neighbours is ', accuracy)


# Desenam graficul de acuratete in functie de nr de vecini
neigh = [i for i in range(3,11)]
predictsGraph = []

for i in neigh:
    predictsGraph.append(knn.getAccuracy(knn.applyClassification(validation_images, i), validation_images_labels))

matpl.plot(neigh, predictsGraph)
matpl.xlabel("number of neighbours")
matpl.ylabel("accuracy")
matpl.show()

# Matricea de confuzie pentru datele de validare
mat_conf = confusion_matrix(validation_images_labels, predictions, normalize='true')
mat_conf = ConfusionMatrixDisplay(mat_conf)
mat_conf.plot()
matpl.show()



# Obtinem predictiile pentru datele de testare
predictTest = knn.applyClassification(test_images, 5)

# Scriem in fisier sub formatul cautat
f = open('submission.txt', 'w')

f.write("id,label\n")
for i in range(len(test_images_names)):
    f.write(f"{test_images_names[i]},{predictTest[i]}\n")

f.close()


finish_time = time.time()   # Pentru a masura cat timp a rulat programul
print(round(finish_time - start_time, 3), 'seconds')
