#-------------------------------------------------------------------
# @author 
# @copyright (C) 2018, 
# @doc
#
# @end
# Created : 21. May 2018 2:41 PM
#-------------------------------------------------------------------

import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import load
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2

class Sample():
    def __init__(self):
        print('Class Created')
        self.load_feature_model()
        self.load_tokenizer()
        self.load_caption_model()

    def load_feature_model(self):
        self.model = VGG16()
        self.model.layers.pop()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)

    def load_tokenizer(self):
        self.tokenizer = load(open("pkl/tokenizer.pkl", 'rb'))
        self.tokenizer.oov_token = None
        self.details = load(open("pkl/detail.pkl", 'rb'))
        self.max_length = self.details['max_length']
        self.vocab_size = self.details['vocab_size']

        self.word_dict = dict()
        for word, index in self.tokenizer.word_index.items():
            self.word_dict[index] = word

    def find_feature(self, image_file_path):
        image = load_img(image_file_path, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = self.model.predict(image, verbose=0)
        return feature

    def load_caption_model(self):
        model_file = 'model/model-ep002-loss3.670-val_loss3.849.h5' # 'model-ep005-loss3.226-val_loss3.783.h5'
        self.caption_model = load_model(model_file)

    def get_work_from_index(self, pred):
        if pred in self.word_dict.keys():
            return self.word_dict[pred]
        else:
            return None

    def get_caption(self, image_file_path):
        feature = self.find_feature(image_file_path)
        initial = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([initial])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            pred = self.caption_model.predict([feature,sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.get_work_from_index(pred)
            if word is None:
                break
            initial += ' ' + word

            if word == 'endseq':
                break
        return initial


if __name__ == '__main__':
    s = Sample()
    image_path = 'test/example.jpg'
    caption = s.get_caption(image_path)
    image = cv2.imread(image_path)

    caption = caption.replace('startseq','')
    caption = caption.replace('endseq', '')

    cv2.putText(image, caption, (10, image.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_4)
    cv2.imwrite("test/example_output.jpg", image)
    cv2.imshow("Image Captioning", image)
    cv2.waitKey(2000)