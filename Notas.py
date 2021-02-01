from tkinter import * 
import tkinter.font as tkFont
import pyaudio
import wave
import os
import time

from modelo import modelo
import tensorflow as tf
import pandas as pd
import numpy as np
import os

class Notas:
    def __init__(self):
        self.App = Tk()
        self.pyaudio = pyaudio.PyAudio()
        self.parametrosGrabacion()
        self.objModelo = modelo()
        
    def main(self):
        self.model = self.loadModel()
        self.Interface()
    
    def Interface(self):
        self.App.geometry("400x250")
        self.App.title("Reconocimiento de Notas Musicales")
        letterStyle = tkFont.Font(family="Lucida Grande", size=15)
        self.labelTitle = Label(self.App, text="Grabar Nota Musical", font=letterStyle, wraplength=600)
        self.labelText = Label(self.App, text="¡Comencemos!")
        self.labelNota = Label(self.App, text="",font=letterStyle,fg="blue")
        self.botonGrabar=Button(self.App, text="¡Grabar Nota!", command=self.grabarNota)
        self.botonReproducir=Button(self.App, text="¡Reproducir Nota!", command=self.reproducirNota)
        self.botonPredecir=Button(self.App, text="¡Predecir!", command= lambda: self.prediccion(self.model))
        self.labelTitle.place(x=100, y=30, width=200, height=30)
        self.labelNota.place(x=100, y=185, width=200, height=30)
        self.labelText.place(x=100, y=60, width=200, height=30)
        self.botonGrabar.place(x=50, y=100, width=145, height=30)
        self.botonReproducir.place(x=205, y=100, width=145, height=30)
        self.botonPredecir.place(x=125, y=140, width=150, height=30)
        self.App.mainloop()

    def parametrosGrabacion(self):
        #Paramatros de Audio

        #Definicion del formato de samples(muestra de sonido)
        self.Format = pyaudio.paInt16
        #Numero de canales
        self.Channels = 1
        #Frecuencia de muestreo 44100 frames por segundo
        self.Rate = 44100
        #Unidades de memoria menores que almacenaran la transmision de datos del flujo de información continuo del sonido, (1024 frames)
        self.Chunk = 1024
        #Duración de la muestra de sonido
        self.Time = 2
        self.contador=0
        
    def grabarNota(self):
        self.stream = self.pyaudio.open(format = self.Format, 
                            channels = self.Channels, 
                            rate = self.Rate, 
                            input = True, 
                            frames_per_buffer = self.Chunk)
        self.labelText.config(text = "Grabando...")
        self.labelText.config(text = "Grabación terminada.")   
        print("Estoy en grabarNota")
        
        #self.contador = self.contador + 1
        #self.Archive = "si"+str(self.contador)+".wav"
        self.Archive = "notaMusical.wav"
        #Inicia proceso de grabado
        #Recolectando información del sonido
        frames = []
        for i in range(0, int(self.Rate/self.Chunk*self.Time)):
            data = self.stream.read(self.Chunk)
            frames.append(data)
        
        #Detención de la grabación
        self.labelText.config(text = "Grabación terminada.", fg="red")    
        #self.stream.stop_stream()
        self.stream.close()
        #self.pyaudio.terminate()

        #Creación y guardado de archivo de audio
        waveFile = wave.open(self.Archive, 'wb')
        waveFile.setnchannels(self.Channels)
        waveFile.setsampwidth(self.pyaudio.get_sample_size(self.Format))
        waveFile.setframerate(self.Rate)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
    def loadModel(self):
        model_path = os.path.join(os.getcwd(), 'models','first_model.hdf5')
        model = tf.keras.models.load_model(model_path)
        return model

    def reproducirNota(self):
        print("Estoy en reproducirNota")
        self.labelText.config(text = "Reproduciendo Audio.")
        Chunk = 1024

        #Ubucación del Archivo
        f = wave.open(r""+os.path.join(os.getcwd(),"notaMusical.wav"),"rb")    

        #Abrir Stream
        stream = self.pyaudio.open(format = self.pyaudio.get_format_from_width(f.getsampwidth()), 
                            channels = f.getnchannels(), 
                            rate = f.getframerate(), 
                            output = True)
        
        #Leyendo Información
        data = f.readframes(Chunk)

        #Reproduciendo Audio
        while data:
            stream.write(data)
            data = f.readframes(Chunk)

        #Parando Stream
        #stream.stop_stream()
        stream.close()
    
        #self.pyaudio.terminate()
              
    def prediccion(self, model):
        pathNota = r""+os.path.join(os.getcwd(),"notaMusical.wav")
        vector_data = []
        
        #Obtengo Caracteristicas
        data = self.objModelo.extraer_caracteristicas(pathNota)
        vector_data.append([data])
         
        #Representación en imagem
        feature = pd.DataFrame(vector_data, columns=['data'])
        print(len(feature))
        
        # Convertir características y etiquetas de clasificación correspondientes en matrices numpy
        X = np.array(feature.data.tolist())
        
        result = model.predict(X)
        indi=0
        vari=result.max()
        for indice, resultado in enumerate(result[0]):
            if resultado == vari:
                indi=indice 
        if(indi==0):
            self.labelNota.config(text = "Do")
        elif(indi==1):
            self.labelNota.config(text = "Re")
        elif(indi==2):
            self.labelNota.config(text = "Fa")
        elif(indi==3):
            self.labelNota.config(text = "La")
        elif(indi==4):
            self.labelNota.config(text = "Si")

            

if __name__ == "__main__":
    notas = Notas()
    notas.main()
    
