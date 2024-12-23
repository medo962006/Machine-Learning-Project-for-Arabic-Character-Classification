import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , Activation , Flatten , Conv2D , MaxPooling2D
from random import shuffle
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from sklearn.preprocessing import StandardScaler
from os import listdir
#from PIL import Image
import cv2
import pickle
training_data = []
"""alphabetBeginEnd = {
    ['ain_begin' , 'ain_end' , 'ain_middle' , 'ain_regular'] : 'ain',
    ['alif_end' , 'alif_hamza' , 'alif_regular'] : 'alif',
    ['beh_begin' , 'beh_end' , 'beh_middle' , 'beh_regular'] : 'beh',
    ['dal_end' , 'dal_regular'] : 'dal',
    ['feh_begin' , 'feh_end' , 'feh_middle' , 'feh_regular'] : 'feh',
    ['heh_begin' , 'heh_end' , 'heh_middle' , 'heh_regular'] : 'heh',
    ['jeem_begin' , 'jeem_end' , 'jeem_middle' , 'jeem_regular'] : 'jeem',
    ['kaf_begin' , 'kaf_end' , 'kaf_middle' , 'kaf_regular'] : 'kaf',
    ['lam_alif'] : 'lam_alif',
    ['lam_begin' , 'lam_end' , 'lam_middle' , 'lam_regular'] : 'lam',
    ['meem_begin' , 'meem_end' , 'meem_middle' , 'meem_regular'] : 'meem',
    ['noon_begin' , 'noon_end' , 'noon_middle' , 'noon_regular'] : 'noon',
    ['qaf_begin' , 'qaf_end' , 'qaf_middle' , 'qaf_regular'] : 'qaf',
    ['raa_end' , 'raa_regular'] : 'raa',
    ['ain_begin' , 'ain_end' , 'ain_regular'] : 'ain',
    ['sad_begin' , 'sad_end' , 'sad_middle' , 'sad_regular'] : 'sad',
    ['seen_begin' , 'seen_end' , 'seen_middle' , 'seen_regular'] : 'seen',
    ['tah_end' , 'tah_middle' , 'tah_regular'] : 'tah',
    ['waw_end' , 'waw_regular'] : 'waw',
    ['yaa_begin' , 'yaa_end' , 'yaa_middle' , 'yaa_regular'] : 'yaa'
    }"""

test_directory = ""
train_directory = ""
categoriesabstract = ['ain','alif', 'beh','dal','feh','heh','jeem','kaf','lam_alif','lam','meem','noon','qaf','raa','sad','seen','tah','waw','yaa']
categories = ['ain_begin' , 'ain_end' , 'ain_middle' , 'ain_regular','alif_end' , 'alif_hamza' , 'alif_regular','beh_begin' , 'beh_end' , 'beh_middle' , 'beh_regular','dal_end' , 'dal_regular','feh_begin' , 'feh_end' , 'feh_middle' , 'feh_regular','heh_begin' , 'heh_end' , 'heh_middle' , 'heh_regular','jeem_begin' , 'jeem_end' , 'jeem_middle' , 'jeem_regular','kaf_begin' , 'kaf_end' , 'kaf_middle' , 'kaf_regular','lam_alif','lam_begin' , 'lam_end' , 'lam_middle' , 'lam_regular','meem_begin' , 'meem_end' , 'meem_middle' , 'meem_regular','noon_begin' , 'noon_end' , 'noon_middle' , 'noon_regular','qaf_begin' , 'qaf_end' , 'qaf_middle' , 'qaf_regular','raa_end' , 'raa_regular','ain_begin' , 'ain_end' , 'ain_regular','sad_begin' , 'sad_end' , 'sad_middle' , 'sad_regular','seen_begin' , 'seen_end' , 'seen_middle' , 'seen_regular','tah_end' , 'tah_middle' , 'tah_regular','waw_end' , 'waw_regular','yaa_begin' , 'yaa_end' , 'yaa_middle' , 'yaa_regular']
directory = './train/isolated_alphabets_per_alphabet'
"""img = tf.keras.preprocessing.image.load_img('./train/isolated_alphabets_per_alphabet/ain_begin/user001_ain_begin_031.png')
train = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)"""
def training_data_initializer():
    counter = 0
    print("Working %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for category in categories:
        path = directory+'/'+category
        Image_category = categories.index(category)
        for img in listdir(path):
            try:
                counter+=1
                if counter % 1000 == 0:
                    print(counter)
                img_array = cv2.imread(f'{path}/{img}', cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array , (64 , 64))
                #plt.imshow(img , cmap = plt.cm.binary)
                training_data.append([new_array , Image_category])
            except Exception as e:
                pass
training_data_initializer()
print(len(training_data))
shuffle(training_data)
X = []
Y = []
for features , label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1 , 64 , 64)
pickle_out = open("X.pickle" , "wb")
pickle.dump(X , pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle" , "wb")
pickle.dump(Y , pickle_out)
pickle_out.close()