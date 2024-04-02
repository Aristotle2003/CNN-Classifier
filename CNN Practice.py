###I downloaded the test and train directory, and each directory has two subdirectories called with-mask and without-mask

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception


base_dir = 'path_to_dataset'

train_dir = f'{base_dir}/train'
test_dir = f'{base_dir}/test'

# Parameters
image_size = {'vgg16': (224, 224), 'resnet': (224, 224), 'inception_v3': (299, 299)}
batch_size = 32

### Model For VGG16
train_datagen_vgg = ImageDataGenerator(preprocessing_function=preprocess_input_vgg) #initialize
train_generator_vgg = train_datagen_vgg.flow_from_directory(
    train_dir,
    target_size=image_size['vgg16'],
    batch_size=batch_size,
    class_mode='binary')

test_datagen_vgg = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
test_generator_vgg = test_datagen_vgg.flow_from_directory(
    test_dir,
    target_size=image_size['vgg16'],
    batch_size=batch_size,
    class_mode='binary')

model_vgg = Sequential([
    VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Model For ResNet

train_datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
train_generator_resnet = train_datagen_resnet.flow_from_directory(
    train_dir,
    target_size=image_size['resnet'],
    batch_size=batch_size,
    class_mode='binary')

test_datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
test_generator_resnet = test_datagen_resnet.flow_from_directory(
    test_dir,
    target_size=image_size['resnet'],
    batch_size=batch_size,
    class_mode='binary')

model_resnet = Sequential([
    ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Model For Inception V3

# Data generators for InceptionV3
train_datagen_inception = ImageDataGenerator(preprocessing_function=preprocess_input_inception)
train_generator_inception = train_datagen_inception.flow_from_directory(
    train_dir,
    target_size=image_size['inception_v3'],
    batch_size=batch_size,
    class_mode='binary')

test_datagen_inception = ImageDataGenerator(preprocessing_function=preprocess_input_inception)
test_generator_inception = test_datagen_inception.flow_from_directory(
    test_dir,
    target_size=image_size['inception_v3'],
    batch_size=batch_size,
    class_mode='binary')

model_inception = Sequential([
    InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling='avg'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_inception.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Train and evaluate three models(use specific model name for each method):
model.fit(train_generator, epochs=10)

loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy*100:.2f}%")
