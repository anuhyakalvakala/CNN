from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten ,MaxPooling2D, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt

print("CNN Architecture-1")

    # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m = Sequential()
m.add(Rescaling(1/255))
m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Conv2D(64, (3, 3), activation='relu'))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Flatten())
m.add(Dense(128, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(5, activation='softmax'))
    # setting and training
m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
# testing
print("Testing.")
score = m.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
m.save("***path***/my_model1.h5")

#########################################################

print("CNN Architecture-2")

    # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m2 = Sequential()
m2.add(Rescaling(1/255))
m2.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m2.add(MaxPooling2D(pool_size=(2, 2)))

m2.add(Flatten())
m2.add(Dense(128, activation='relu'))
m2.add(Dropout(0.5))
m2.add(Dense(5, activation='softmax'))
    # setting and training
m2.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m2.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m2.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
###################################################
print("CNN Architecture-3")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m1 = Sequential()
m1.add(Rescaling(1/255))
m1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m1.add(MaxPooling2D(pool_size=(2, 2)))
m1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m1.add(MaxPooling2D(pool_size=(2, 2)))
m1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
m1.add(MaxPooling2D(pool_size=(2, 2)))
m1.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
m1.add(MaxPooling2D(pool_size=(2, 2)))
m1.add(Flatten())
m1.add(Dense(128, activation='relu'))

m1.add(Dense(5, activation='softmax'))
    # setting and training
m1.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m1.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m1.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
###################################################
print("CNN Architecture-4")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m3 = Sequential()
m3.add(Rescaling(1/255))
m3.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m3.add(MaxPooling2D(pool_size=(2, 2)))
m3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m3.add(MaxPooling2D(pool_size=(2, 2)))
m3.add(Flatten())
m3.add(Dense(128, activation='relu'))
m3.add(Dropout(0.5))
m3.add(Dense(5, activation='softmax'))
    # setting and training
m3.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m3.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m3.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])

###################################################
print("CNN Architecture-5")
""" Trains and evaluates CNN image classifier on the flowers dataset.
        Returns the trained model. """
    # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m4 = Sequential()
m4.add(Rescaling(1/255))
m4.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

m4.add(Flatten())
m4.add(Dense(128, activation='relu'))
m4.add(Dropout(0.5))
m4.add(Dense(5, activation='softmax'))
    # setting and training
m4.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m4.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m4.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
###################################################
print("CNN Architecture-6")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m5 = Sequential()
m5.add(Rescaling(1/255))
m5.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m5.add(MaxPooling2D(pool_size=(2, 2)))
m5.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m5.add(MaxPooling2D(pool_size=(2, 2)))
m5.add(Flatten())
m5.add(Dense(128, activation='relu'))
m5.add(Dropout(0.5))
m5.add(Dense(5, activation='softmax'))
    # setting and training
m5.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m5.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m5.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])

###################################################
print("CNN Architecture-7")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m6 = Sequential()
m6.add(Rescaling(1/255))
m6.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m6.add(MaxPooling2D(pool_size=(2, 2)))
m6.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m6.add(MaxPooling2D(pool_size=(2, 2)))
m6.add(Flatten())
m6.add(Dense(128, activation='relu'))

m6.add(Dense(5, activation='softmax'))
    # setting and training
m6.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m6.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m6.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])

###################################################
print("CNN Architecture-8")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m7 = Sequential()
m7.add(Rescaling(1/255))
m7.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m7.add(MaxPooling2D(pool_size=(2, 2)))
m7.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m7.add(MaxPooling2D(pool_size=(2, 2)))
m7.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
m7.add(MaxPooling2D(pool_size=(2, 2)))
m7.add(Flatten())
m7.add(Dense(128, activation='relu'))
m7.add(Dropout(0.5))
m7.add(Dense(5, activation='softmax'))
    # setting and training
m7.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m7.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m7.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])

###################################################
print("CNN Architecture-9")
  # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    # build the model
m8 = Sequential()
m8.add(Rescaling(1/255))
m8.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,3)))
m8.add(MaxPooling2D(pool_size=(2, 2)))
m8.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
m8.add(MaxPooling2D(pool_size=(2, 2)))
m8.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
m8.add(MaxPooling2D(pool_size=(2, 2)))
m8.add(Flatten())
m8.add(Dense(128, activation='relu'))
m8.add(Dropout(0.5))
m8.add(Dense(5, activation='softmax'))
    # setting and training
m8.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m8.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
    # testing
print("Testing.")
score = m8.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])

###################################################
print("CNN Architecture-10")
    # load datasets
training_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(40,40))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              
label_mode="categorical",
                                                              seed=0,
                                                              image_size=(40,40))
    # build the model
m9 = Sequential()
m9.add(Rescaling(1/255))
m9.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(40,40,3)))
m9.add(MaxPooling2D(pool_size=(2, 2)))
m9.add(Conv2D(64, (3, 3), activation='relu'))
m9.add(MaxPooling2D(pool_size=(2, 2)))
m9.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
m9.add(MaxPooling2D(pool_size=(2, 2)))
m9.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
m9.add(MaxPooling2D(pool_size=(2, 2)))
m9.add(Flatten())
m9.add(Dense(128, activation='relu'))
m9.add(Dropout(0.5))
m9.add(Dense(5, activation='softmax'))
    # setting and training
m9.compile(loss="categorical_crossentropy", metrics=['accuracy'])
history  = m9.fit(training_set, batch_size=32, epochs=25,verbose=0)
print(history.history["accuracy"])
print(training_set.class_names)
# testing
print("Testing.")
score = m9.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
###################################################