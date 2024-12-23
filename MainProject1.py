import cv2
import PIL.Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle
from os import listdir



#Image Preprocessing & Data Augmentation
training_data = []
categories = ['ain_begin' , 'ain_end' , 'ain_middle' , 'ain_regular','alif_end' , 'alif_hamza' , 'alif_regular','beh_begin' , 'beh_end' , 'beh_middle' , 'beh_regular','dal_end' , 'dal_regular','feh_begin' , 'feh_end' , 'feh_middle' , 'feh_regular','heh_begin' , 'heh_end' , 'heh_middle' , 'heh_regular','jeem_begin' , 'jeem_end' , 'jeem_middle' , 'jeem_regular','kaf_begin' , 'kaf_end' , 'kaf_middle' , 'kaf_regular','lam_alif','lam_begin' , 'lam_end' , 'lam_middle' , 'lam_regular','meem_begin' , 'meem_end' , 'meem_middle' , 'meem_regular','noon_begin' , 'noon_end' , 'noon_middle' , 'noon_regular','qaf_begin' , 'qaf_end' , 'qaf_middle' , 'qaf_regular','raa_end' , 'raa_regular','ain_begin' , 'ain_end' , 'ain_regular','sad_begin' , 'sad_end' , 'sad_middle' , 'sad_regular','seen_begin' , 'seen_end' , 'seen_middle' , 'seen_regular','tah_end' , 'tah_middle' , 'tah_regular','waw_end' , 'waw_regular','yaa_begin' , 'yaa_end' , 'yaa_middle' , 'yaa_regular']
test_directory = ""
train_directory = ""
directory = './train/isolated_alphabets_per_alphabet'
Submission_File = 'Sample Submission File.csv'
counter = 0

for category in categories:
    path = directory+'/'+category
    Image_category = categories.index(category)
    for img in listdir(path):
        try:
            counter+=1
            if counter % 1000 == 0:
                print(counter)
            img_array = cv2.imread(f'{path}/{img}', cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array , (32 , 32))
            training_data.append([new_array , Image_category])
        except Exception as e:
            pass
shuffle(training_data)


#Splitting and preprocessing the dataset to the training data and the testing data
X = []
Y = []
for features , label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1 , 32 , 32, 1)
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.1 , random_state = 0)
x_train = tf.stack(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.stack(x_test)
y_test = tf.convert_to_tensor(y_test)

#Generating Train and Test Datasets for fitting
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(16)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(16)

#Convolutional Neural Network (Seperated as independent layers)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32 , 32 , 1)))
model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=120, activation='relu'))

model.add(tf.keras.layers.Dense(units=84, activation='relu'))

model.add(tf.keras.layers.Dense(units=68, activation = 'softmax'))

model.summary()

#Adjusting the optimal learning rate, compiling the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0005) #0.0001 DEFAULT
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

#Setting Learning Rate Parameters , Fitting The Model
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.00001
)
model.fit(train_ds, 
          validation_data= test_ds, 
          epochs = 80,
          callbacks=[reduce_lr])

#Evaluating the Model (for   'Val_loss' & 'Val_acc'  )
score = model.evaluate(x_test,y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Reading the CSV submission_File
df = pd.read_csv(Submission_File)


#Extracting The Images from the Test Folder
cnt = 0
prediction_data = []
for category in categories:
        path = './test/test/'+category

        for img in listdir(path):
                img_array = cv2.imread(f'{path}/{img}' , cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array , (32 , 32))
                prediction_data.append(new_array)
prediction_data = np.array(prediction_data).reshape(-1 , 32 , 32 ,1)

#defining a function for Label processing
def suffix_decision(suffix):                              
    if suffix.endswith('end'):
        return suffix.replace('_end','')
    elif suffix.endswith('begin'):
        return suffix.replace('_begin','')
    elif suffix.endswith('regular'):
        return suffix.replace('_regular','')
    elif suffix.endswith('middle'):
        return suffix.replace('_middle' , '')
    elif suffix.endswith('hamza'):
        return suffix.replace('_hamza' , '')
    else:
        return suffix

#Predicting New Labels
prediction = model.predict(prediction_data)

#Applying New Labels to the Submission File
y_predict = 0
long_letter = ""
desired_letter = ""
for i in range(13000):
    y_predict=np.argmax(prediction[i])
    long_letter = categories[y_predict]
    desired_letter = suffix_decision(long_letter)
    df.loc[i , 'Letter'] = desired_letter
df.to_csv(Submission_File , index = False)