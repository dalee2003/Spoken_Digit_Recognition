import numpy as np
import librosa.display
import librosa.feature
from keras import models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import models


def extract_mfcc(file, fmax, nMel):
    audioFile = "./Audio_Data16/" + file[0] + "/" + file
    y, sr = librosa.load(audioFile)
    
    plt.figure(figsize=(3, 3), dpi=100)
#    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
#    librosa.display.specshow(D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    #librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), fmax=fmax)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), fmax=fmax)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    #plt.savefig('tmp/tmp/myImg.jpg', bbox_inches='tight', pad_inches=-0.1)

    #pngFileName = "codes/spoken_numbers_wav/mfcc_image_ts/" + file[0] + "/" + file[:-3] + "png"
    pngFileName = "codes/tmp/tmp/" + file[:-3] + "png"
    plt.savefig(pngFileName, bbox_inches='tight', pad_inches=-0.1)

    plt.close()

    
    return

#add to train data
# for i in range(10):
#     for j in range(1,49):
#         for k in range(50):
#             file = str(i) + "_"
#             if j<10:
#                 file += "0" + str(j)
#             else: 
#                 file += str(j)
#             file += "_" + str(k) + ".wav"
#             #print("Audio_Data16/" + file[0] + "/" + file)
#             #print("codes/spoken_numbers_wav/mfcc_image_tr/" + file[0] + "/" + file[:-3] + "png")
#             extract_mfcc(file, 8000, 256)

#add to validation data
# for i in range(10):
#     for j in range(49,55):
#         for k in range(50):
#             file = str(i) + "_"
#             if j<10:
#                 file += "0" + str(j)
#             else: 
#                 file += str(j)
#             file += "_" + str(k) + ".wav"
#             #print("Audio_Data16/" + file[0] + "/" + file)
#             #print("codes/spoken_numbers_wav/mfcc_image_ts/" + file[0] + "/" + file[:-3] + "png")
#             extract_mfcc(file, 8000, 256)

#add to test data
# for i in range(10):
#     for j in range(55,61):
#         for k in range(50):
#             file = str(i) + "_"
#             if j<10:
#                 file += "0" + str(j)
#             else: 
#                 file += str(j)
#             file += "_" + str(k) + ".wav"
#             #print("Audio_Data16/" + file[0] + "/" + file)
#             #print("codes/tmp/tmp/" + file[:-3] + "png")
#             extract_mfcc(file, 8000, 256)
nrow = 200
ncol = 200


model = models.load_model('mfcc_cnn_model.h5')

def predict():
    file = "7_60_19.wav"
    extract_mfcc(file, 8000, 256)
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False)
    test_generator = test_datagen.flow_from_directory(
            './codes/tmp',
            target_size=(nrow, ncol),
            batch_size=1,
            class_mode='sparse')

    # Load the model
    #Xts, _ = test_generator.next()
    Xts, _ = next(test_generator)
    print("_", _)
    print("Xts",Xts)



    # Predict the probability of each class
    yts = model.predict(Xts)
    print ("yts*100",yts * 100)
    if np.max(yts) < 0.1:
        print ('Cannot Recognize!')
        #s1.set('Cannot Recognize!')
        #top.update()
        return

    # Choose the most likely class
    res = np.argmax(yts)
    #print (res)
    print("You said "+ str(res))
    #s1.set('You said: '+ str(res))
    #top.update()
    return

predict()
