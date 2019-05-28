from globals import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adadelta
import matplotlib.pyplot as plt 
from keras.callbacks import Callback
from keras.layers import *
import os


''' Builds an ANN with similar archetecture to 
    https://medium.com/nanonets/how-to-easily-build-a-dog-breed-image-classification-model-2fd214419cde

    with some differences:
        Notably the conv layer sizes, and the additional fully connected layers

    Returns an untrained model built for our data
'''
def build_net(inputShape):
    cnn = Sequential()

    # Conv 1 to look at big chunks
    cnn.add(Conv2D(32, (10,10), strides=3, input_shape=inputShape))
    cnn.add(AveragePooling2D())
    cnn.add(Activation('relu'))

    # Conv 1 to look at semi big chunks
    cnn.add(Conv2D(32, (5,5)))
    cnn.add(AveragePooling2D())
    cnn.add(Activation('relu'))

    # Conv 2 to look at smaller chunks
    cnn.add(Conv2D(64, (4,4)))
    cnn.add(AveragePooling2D())
    cnn.add(Activation('relu')) 

    # Conv 3 to look at smallest chunks
    cnn.add(Conv2D(128, (3,3)))
    cnn.add(AveragePooling2D())
    cnn.add(Activation('relu'))


    # Then do fully connected with lots of nodes
    cnn.add(Flatten())

    cnn.add(Dense(500))
    cnn.add(Activation('relu'))

    cnn.add(Dense(300))
    cnn.add(Activation('relu'))

    cnn.add(Dense(150))
    cnn.add(Activation('relu'))
    

    # Then fully connect to output
    cnn.add(Dense(2))
    cnn.add(Activation('softmax'))

    cnn.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    return cnn

''' Train the model on the training set of data
    Keras is really good about separating the test data so we're sure the
    model isn't going to be influenced by images it has never seen before.

    Really all we have to do is sit back and let the model train
'''
def train():
    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        zoom_range=[0,0.2],
        channel_shift_range=0.1,
    )

    test_gen = ImageDataGenerator()

    train_flow = train_gen.flow_from_directory(
        TRAIN, batch_size=BATCH, 
        shuffle=True, 
        target_size=(DIMS,DIMS)
    )

    test_flow = test_gen.flow_from_directory(
        TEST, 
        batch_size=BATCH, 
        shuffle=True, 
        target_size=(DIMS,DIMS)
    )

    cnn = build_net((DIMS,DIMS,3))

    try:
        history = cnn.fit_generator(
            train_flow,
            validation_data=test_flow,
            steps_per_epoch=NUM_TRAIN // BATCH,
            validation_steps=NUM_TEST // BATCH,
            epochs=EPOCHS,
        )
    except KeyboardInterrupt:
        pass

    cnn.save_weights(CNN_WEIGHTS)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss for Abstract v. Renaissance')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


''' Loads the model with the best weights found so far to validate what we (hopefully)
    already know: their accuracy.

    May change slightly from when it was training in the model, but should be just about 
    whatever the training section says it is.

    Used to prove the models correctness. Can also be run on other directories the model
    has never seen before just for fun
'''
def run(test_dir=TEST):
    cnn = build_net((DIMS, DIMS, 3))
    cnn.load_weights(CNN_BEST)

    gen = ImageDataGenerator()
    test_flow = gen.flow_from_directory(test_dir, batch_size=1, target_size=(DIMS,DIMS))
    
    correct = 0
    for i in range(len(test_flow)):
        x,y = test_flow[i]
        p = cnn.predict_on_batch(x)
        
        if p[0].argmax() == y[0].argmax():
            correct += 1
            print("Correct")

        else:
            print("Incorrect: ", end='')
            print("Predicted: " + str(p[0]) + "\tActual: " + str(y[0]))

    print("Accuracy: " + str(correct/len(test_flow))) 
    return(correct/len(test_flow))


#train()
run()