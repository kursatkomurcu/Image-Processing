from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
directory = os.chdir('E:\\NutzenTech\\Dataset\\Elektrik Direkleri\\insulator_defective\\images')

for i in os.listdir(directory):
    img = load_img(i)
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    j = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='E:\\NutzenTech\\Dataset\\Elektrik Direkleri\\insulator_defective\\augmentation',
                              save_format='jpg'):
        j +=1
        if j > 10:
            break