import numpy as np
from matplotlib import pyplot as matpl
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.python.keras import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Pentru acest model nu mai aplicam 'flatten' pe imagini ; stie sa lucreze cu imagini 2D

# Citirea si prelucrarea fisierului train.txt
train_aux = np.loadtxt('./train.txt', dtype='str')[1:]
train_aux = [x.split(',') for x in train_aux]
train_images_names = [x[0] for x in train_aux]
train_images_labels = np.array([int(x[1]) for x in train_aux])

# Citirea fisierului test.txt
test_images_names = np.loadtxt('./test.txt', dtype='str')[1:]

# Citirea si prelucrarea fisierului validation.txt
validation_aux = np.loadtxt('./validation.txt', dtype='str')[1:]
validation_aux = [x.split(',') for x in validation_aux]
validation_images_names = [x[0] for x in validation_aux]
validation_images_labels = np.array([int(x[1]) for x in validation_aux])

# citim imaginile cu imread din libraria matplotlib
train_images = np.array([matpl.imread(f"train+validation/{name}") for name in train_images_names])

# citim imaginile de validare
validation_images = np.array([matpl.imread(f"train+validation/{name}") for name in validation_images_names])

# citim imaginile de test
test_images = np.array([matpl.imread(f"test/{name}") for name in test_images_names])

# Definim modelul CNN ca fiind un keras.Sequential in care adaugam manual apoi layerele
model = Sequential(
    # Un prim layer convolutional cu 100 de filtre si un kernel de (3,3);
    # strides este default de (1,1); padding = same ne asigura ca imaginea isi pastreaza
    # dimensiunea in layerul urmator; folosim functia liniara de activare
    [Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(16, 16, 3)),
     # Este bun un dropout minim pentru a evita overfittingul atunci cand folosim multiple straturi convolutionale
     Dropout(rate=0.1),

     # Un al doilea strat convolutional cu 150 de filtre; crestem numarul de filtre de la strat la strat pentru a
     # delimita cat mai bine featureurile
     Conv2D(filters=150, kernel_size=(3, 3), activation='relu', padding='same'),
     # Pooling ce injumatateste practic (fiind (2,2)) dimensiunile imaginii
     MaxPool2D(pool_size=(2, 2), strides=2),
     # Dropout unui sfert de neuroni pentru a evita overfittingul
     Dropout(rate=0.3),

     # Un al treilea strat convolutional. De data aceasta folosim un kernel de (2,2) fiindca imaginea a ramas destul de
     # mica, si asa gasim mai precis featureurile
     Conv2D(filters=200, kernel_size=(2, 2), activation='relu', padding='same'),
     MaxPool2D(pool_size=(2, 2), strides=2),
     Dropout(rate=0.2),

     # Dupa doua pooling-uri, imaginea a ajuns 4x4. Aplicam flatten si obtinem 16x1 pe care o pasam mai departe
     Flatten(),

     # O serie de Straturi complete urmate de dropout(pentru a evita oferfitting) din care modelul invata
     Dense(units=300, activation='relu'),
     Dropout(rate=0.3),
     # De la strat la strat avem mai putine unitati pentru a ne apropia predictia cat mai mult de raspuns
     Dense(units=100, activation='relu'),
     Dropout(rate=0.3),

     # Softmax este cel mai bun activation pentru clasificarea imaginilor in mai mult de 2 clase
     Dense(units=7, activation='softmax')]
)

# Compilam modelul
# Adam este un optimizer bun pentru probleme de clasificare. Adamax mai scoate valori bune
# metrics = 'accuracy' imi afiseaza in consola acuratetea in fiecare epoca
# Sparse categorical crossentropy este un loss function probabilistic ce accepta ca labels-urile sa fie integers
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Antrenam modelul si ii pasam prin parametri si datele de validare
# Pe layerele construite am obtinut valori optime in intervalul 13-16 epoci
modelfit = model.fit(train_images, train_images_labels, epochs=15, validation_data=(validation_images, validation_images_labels))

# Obtinem in consola o descriere a modelului, in special o listare a layerelor si parametrilor per layer
model.summary()

# Prezicem imaginile de testare
predicts = model.predict(test_images)

# Trasam graficul acuratetii in functie de numarul de epoci utilizand istoricul modelului
matpl.plot(modelfit.history['val_accuracy'])
matpl.xlabel("epoca")
matpl.ylabel("val_accuracy")
matpl.show()

# Matricea de confuzie pentru datele de validare
# Hardcoded predictsValid pentru a obtine matricea de confuzie mai usor
predictsValid = model.predict(validation_images)
predictsValid = [x.argmax() for x in predictsValid]

mat_conf = confusion_matrix(validation_images_labels, predictsValid, normalize='true')
mat_conf = ConfusionMatrixDisplay(mat_conf)
mat_conf.plot()
matpl.show()


# Afisarea in fisier sub formatul dorit
f = open('submission3.txt', 'w')

f.write("id,label\n")
for i in range(len(predicts)):
    f.write(f"{test_images_names[i]},{predicts[i].argmax()}\n")

f.close()
