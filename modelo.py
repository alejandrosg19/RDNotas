import numpy as np
import pandas as pd
import os
import librosa

#extraccion de caracteristicas
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#Division de datos
from sklearn.model_selection import train_test_split 

#Modelo
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

#Entrenando Modelo
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

class modelo():
    
    def __init__(self):
        print("hola")
     
    def main(self):
        #cargo el data 
        data = self.loadData()
        #Convierto datos categoricos a numeros
        X,yy = self.convertiendo_data(data)
        #Conjunto de datos de entrenamiento y testeo
        x_train, x_test, y_train, y_test  = self.entranamiento_test(X, yy)
        #Creación y compilación del modelo
        model = self.modelo(yy,x_test, y_test)
        #Entrenamiento del modelo
        finalModel = self.entranamiento_Modelo(model,x_train, x_test, y_train, y_test)
        #Evaluar Modelo
        self.evaluar_Modelo(finalModel, x_test, y_test)
    
    #cargamos datos y damos una clasificación
    def loadData(self):
        etiqueta = 0

        features = []
        #Se obtiene el directorio actual y se mueve al dataset    
        data_folder = os.path.join(os.getcwd(), "dataset")
        #Listo cada carpeta dentro del dataset
        folders = os.listdir(data_folder)

        #Iteramos cada carpeta del dataset
        for folder in folders:
            #Tengo la ubicación del dataset y me muevo a la dirección de cada carpeta
            dir_folder = os.path.join(data_folder, folder)

            #listo las notas musicales
            notasMusicales = os.listdir(dir_folder)

            #Itero por cada uno de los audios de lsa notas musicales
            for nota in notasMusicales:
                #clasificamos en etiqueta
                if "do" in nota:
                    etiqueta = 1
                elif "re" in nota:
                    etiqueta = 2
                elif "fa" in nota:
                    etiqueta = 3
                elif "la" in nota:
                    etiqueta = 4 
                elif "si" in nota:
                    etiqueta = 5
                else:
                    etiqueta = "N/A"               
                
                #Obtengo el directorio de la nota
                pathNota = os.path.join(dir_folder, nota) 
                #obtengo caracteristicas del audio
                data = self.extraer_caracteristicas(pathNota)
                #genero un vector con el data y su etiqueta
                features.append([data, etiqueta])
        
        #Con pandas almaceno la representación de la imagen del audio en un dataframe junto con su etiqueta
        featuresdf = pd.DataFrame(features, columns=['feature','etiqueta'])
        print('Finalizando extracción de caracteristicas de ', len(featuresdf), ' filas')     
        
        #features = featuresdf.loc[1]
        #print( list(features) )    
        return featuresdf
                
    #extraigo caracteristicas propias de cada imagen haciendo uso de librosa
    #extraigo caracteristicas de epectrgrama de la imagen
    def extraer_caracteristicas(self,file_name):
        try:
            #la funcion mfcc de librosa me permite un MFCC a partir de los datos del audio
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception as e:
            print("Error no encontre la dirección de audio: ", file_name)
            return None 
        
        return mfccsscaled
    
    #Es necesario convertir los datos categoricos a numericos, para eso uso LabelEnconder, 
    #esto para que el modelo pueda entenerlo
    def convertiendo_data(self, featuresdf):
        # Convertir características y etiquetas de clasificación correspondientes en matrices numpy
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.etiqueta.tolist())

        #Codifico las etiquetas de clasificacipon
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y)) 
        
        print(X)
        print(y)
        print(yy)
        
        return X,yy
    
    #Divido el conjunto de datos en dos bloques 80% entrenamiento 20% test 
    #saco valores de X Y
    def entranamiento_test(self, X, yy):
        #Divido el conjunto de datos
        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
        
        #Conjunto de datos de entrenamiento
        print(x_train.shape)
        #Conjunto de datos Test
        print(x_test.shape)
        
        return x_train, x_test, y_train, y_test 
     
    #Se crean una red neuronal mediante un perceptron multicapa (MLP) usando keras y tensorflow
    #Se plantea un modelo secuencial para construir el modelo capa por capa
    #Capas:
    #CAPA DE ENTRADA: 40 nodos, la función MFCC de extracción de caracteristicas devuelve un conjunto de datos 1x40
    #CAPA OCULTA de 256 nodos: estas capas tendrán una capa densa con una función de activación de tipo "ReLu", 
    # (se ha demostrado que esta función de activación funciona bien en redes neuronales). 
    # También destacar que aplicaremos un valor de abandono del 50% en nuestras dos primeras capas. 
    # Esto excluirá al azar los nodos de cada ciclo de actualización, lo que a su vez da como resultado una 
    # red que es capaz de respondr mejor a la generalización y es menos probable se produzca sobreajuste los datos de entrenamiento.    
    #CAPA SALIDA 5 nodos: que coinciden con el número de clasificaciones posibles. 
    # La activación es para nuestra capa de salida es softmax. Softmax hace que la salida sume 1, 
    # por lo que la salida puede interpretarse como probabilidades. 
    # El modelo hará su predicción según la opción que tenga la mayor probabilidad. 
    def modelo(self, yy,x_test, y_test):
        num_labels = yy.shape[1]
        filter_size = 2

        # Construcción del Modelo
        model = Sequential()

        model.add(Dense(256, input_shape=(40,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
        
        #Compilando Modelo. para compilar el modelo se usaran los siguientes parametros:
        #*Función de Perdida: utilizaremos categorical_crossentropy. Esta es la opción más común para la clasificación. 
        # Una puntuación más baja indica que el modelo está funcionando mejor.
        #*Métricas: utilizaremos la métrica de accuracy que nos permitirá ver el puntaje de precisión en los datos de validación cuando entrenemos el modelo.
        #*Optimizador: aquí usaremos adam, que generalmente es un buen optimizador para muchos casos de uso.
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 
        
        #Muestro resumen de la arquitectura del modelo
        model.summary()
        
        #Calculo la precisión previa al entrenamiento
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100*score[1]
        
        print("Pre-training accuracy: %.4f%%" % accuracy)
        
        return model
    
    #ENTRANDO MODELO
    #Se empieza probando con un número de epocas baja y se prueba hasta ver donde alcanza un valor asintotico donde por más 
    # que subamos las epocas no conseguimos que el modelo mejore significativamente.
    # Por otro lado, el tamaño del lote debe ser suficientemente bajo, ya que tener un tamaño de lote grande puede 
    # reducir la capacidad de generalización del modelo
    def entranamiento_Modelo(self, model,x_train, x_test, y_train, y_test):
        num_epochs = 100
        num_batch_size = 32
        
        checkpointer = ModelCheckpoint(filepath='models/first_model.hdf5', 
                               verbose=1, save_best_only=True)
        start = datetime.now()
        model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, 
          validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
        
        duration = datetime.now() - start
        print("Training completed in time: ", duration)
        
        return model
    
    def evaluar_Modelo(self, finalModel, x_test, y_test):    
        score = finalModel.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])
    
        
if __name__=="__main__":
    newModel = modelo()
    newModel.main()
                