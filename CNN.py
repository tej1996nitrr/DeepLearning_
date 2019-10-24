from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from  keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen =ImageDataGenerator(rescale=1.0/255)
train_set = train_datagen.flow_from_directory(
r'F:\Jupyter\CATS_DOGS\CATS_DOGS\train',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
r'F:\Jupyter\CATS_DOGS\CATS_DOGS\test',
target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

classifier.fit_generator(train_set,
                         samples_per_epoch=8000,
                         validation_data=test_set,
                         )
results=classifier.fit_generator(train_set,epochs=1,steps_per_epoch=150,validation_data=test_set,validation_steps=12)
