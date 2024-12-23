# %%
import numpy as np
import tensorflow as tf
from os import listdir
from random import shuffle
import cv2

# %%
training_data = []
ok = False
categories = ['Grass' , 'Asphalt' , 'Gravel' , 'Glass' , 'Bricks']
test_directory = ""
train_directory = ""
#Texture images should be inside a folder named train where every category is under a folder with its name, for Training/Evaluation purposes 
directory = './train/TextureImages' # SHOULD BE CHANGED ACCORDING TO THE FOLDER NAME
def training_data_initializer():
    for category in categories:
        path = directory+'/'+category
        Image_category = categories.index(category)
        for img in listdir(path):
            try:
                img_array = cv2.imread(f'{path}/{img}')
                new_array = cv2.resize(img_array , (32 , 32))
                if([new_array,Image_category] not in training_data):
                    training_data.append([new_array , Image_category])
            except Exception as e:
                pass
training_data_initializer()

shuffle(training_data)


# %%
X = []
Y = []
for features , label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1 , 32 , 32, 3)

# %%
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)
x_train = tf.stack(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.stack(x_test)
y_test = tf.convert_to_tensor(y_test)

# %%
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(16)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(16)

# %%
#For Adding new Datasets:-
new_dataset_Path = ""
new_training_data = []
def training_data_initializer():
    if new_dataset_Path != "":
        for category in categories:
            path = new_dataset_Path
            Image_category = categories.index(category)
            for img in listdir(path):
                try:
                    img_array = cv2.imread(f'{path}/{img}')
                    new_array = cv2.resize(img_array , (32 , 32))
                    if([new_array,Image_category] not in training_data):
                        training_data.append([new_array , Image_category])
                except Exception as e:
                    pass
training_data_initializer()

# %%
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32 , 32 , 3)))
model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=120, activation='relu'))

model.add(tf.keras.layers.Dense(units=84, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(units=68, activation = 'softmax'))


# %%
model.summary()

# %%

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.00001
)

# %%
model.fit(train_ds, 
          validation_data= test_ds, 
          epochs = 80,
          callbacks=[reduce_lr])

# %%
score = model.evaluate(x_test,y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("model.h5")


