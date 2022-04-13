import numpy as np
import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfdata
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as matpl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


# Cream un Support vector classifier (SVC) avand kernel liniar si C = 1, valoare destul de mare, cu risc de overfitting,
# Cea mai buna acuratete se obtine pe C = 0.1 cu kernel liniar
SVC = svm.SVC(C=1, kernel="poly")

# Antrenam modelul SVM
SVC.fit(train_images, train_images_labels)

# Realizam predictiile pe validation_images si apoi calculam acuratetea cu functia accuracy_score
predicts = SVC.predict(validation_images)
print('acuratete:', accuracy_score(predicts, validation_images_labels))


# Graficul de eficienta pentru kernel liniar:
# Curi = [float(i/10) for i in range(1, 11)]
# predictsGraph = []
#
# for i in Curi:
#     SVCaux = svm.SVC(C=i, kernel='linear')
#     SVCaux.fit(train_images, train_images_labels)
#     predictsGraph.append(accuracy_score(SVCaux.predict(validation_images), validation_images_labels))
#
# matpl.plot(Curi, predictsGraph)
# matpl.title("linear")
# matpl.xlabel("C")
# matpl.ylabel("accuracy")
# matpl.show()


# Graficul de eficienta pentru kernel poly:
# Curi2 = [float(i/10) for i in range(1, 11)]
# predictsGraph2 = []
#
# for i in Curi2:
#     SVCaux = svm.SVC(C=i, kernel='poly')
#     SVCaux.fit(train_images, train_images_labels)
#     predictsGraph2.append(accuracy_score(SVCaux.predict(validation_images), validation_images_labels))
#
# matpl.plot(Curi2, predictsGraph2)
# matpl.title("poly")
# matpl.xlabel("C")
# matpl.ylabel("accuracy")
# matpl.show()

# Matricea de confuzie pentru datele de validare
mat_conf = confusion_matrix(validation_images_labels, predicts, normalize='true')
mat_conf = ConfusionMatrixDisplay(mat_conf)
mat_conf.plot()
matpl.show()

# Realizam predictiile pentru datele de testare
predictiiTest = SVC.predict(test_images)


# Scriem in fisier sub formatul dorit
f = open('submission2.txt', 'w')

f.write("id,label\n")
for i in range(len(test_images_names)):
    f.write(f"{test_images_names[i]},{predictiiTest[i]}\n")

f.close()
