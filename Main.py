#import numpy as np
#from random import shuffle
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from sklearn.preprocessing import StandardScaler
#from os import listdir
#from PIL import Image
#import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

# Load the data
x = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("Y.pickle", "rb"))

# Preprocess the data
x = x / 255.0  # Normalize the pixel values to [0, 1]
y = np.array(y)  # Convert y to a numpy array if it's not already
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))  # Adding dropout for regularization

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=32, epochs=1, validation_split=0.1)
"""for directory_index in listdir(directory):
    train_dataset = train.flow_from_directory(
        f'{directory}/{directory_index}',
        batch_size = 32,
        target_size = (100 , 100),
        class_mode= 'categorical',
        follow_links=True),
    validation_dataset = validation.flow_from_directory(
        f'./test/{directory_index}'
        batch_size = 32,
        target_size = (100 , 100),
        class_mode = 'categorical',
        follow_links=True)
    """














alphabetFirst3Letters = {
    'ain': 'ain',
    'ali': 'alif', 
    'beh': 'beh',
    'dal': 'dal',
    'feh': 'feh',
    'heh': 'heh',
    'jee': 'jeem',
    'kaf': 'kaf',
    'lam_alif': 'lam_alif',
    'lam': 'lam',
    'mee': 'meem',
    'noo': 'noon',
    'qaf': 'qaf',
    'raa': 'raa',
    'sad': 'sad',
    'see': 'seen',
    'tah': 'tah',
    'waw': 'waw',
    'yaa': 'yaa',
}
"""for i in listdir(directory):
    if(i == 'lam_alif'):
        for x in listdir('./train/isolated_alphabets_per_alphabet/lam_alif'):
            img = Image.open(f'{directory}/lam_alif/{x}').convert("L")
           # img.thumbnail((64 , 64))
            img = np.invert(np.array(img))
            plt.imshow(img , cmap = plt.cm.binary)
            plt.show()
"""