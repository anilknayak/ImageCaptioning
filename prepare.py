#-------------------------------------------------------------------
# @author 
# @copyright (C) 2018, 
# @doc
#
# @end
# Created : 19. May 2018 8:09 PM
#-------------------------------------------------------------------

import os
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import pickle
import string
from pickle import load
from keras.preprocessing.text import Tokenizer
from numpy import array

class Prepare():
    def __init__(self,  sample=False,
                        model_name='VGG16',
                        image_dir='data/images',
                        caption_file='data/caption/Flickr8k.token.txt',
                        clan_caption=False,
                        load_pickle=False):
        self.features = {}
        self.images = {}
        self.description = {}
        self.sample = sample

        self.prepare_model(model_name)

        if load_pickle:
            print('loading from pickle file')
            desc = load(open("description.pkl", 'rb'))
            feature = load(open("features.pkl", 'rb'))
            self.features = feature
            self.description = desc
        else:
            self.load_image_data(image_dir)
            self.find_features()
            self.load_captions(caption_file)
            self.cleaning_captions(clan_caption)
            self.vocabulary()
            self.enumurate_vocab_char_map()
            self.save_feature_and_description()

    def prepare_model(self, filename):
        print('Loading model ', filename)
        self.model = VGG16()
        self.model.layers.pop()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)
        print(self.model.summary())

    def load_image_data(self, directory):
        print('Loading Images from dir ', directory)
        image_file_names = os.listdir(directory)
        count = 0
        for i in tqdm(range(len(image_file_names)), desc="Reading Images"):
            image_file = image_file_names[i]
            if not "jpg" in image_file:
                continue
            filename = os.path.join(directory,image_file)
            image = load_img(filename, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            image_id = image_file.split('.')[0]
            self.images[image_id] = image
            count += 1
            if self.sample and count == 10:
                break

    def find_features(self):
        print('Finding Features for each image loaded ')
        count = 0
        for name in self.images.keys():
            count +=1
            image = self.images[name]
            feature = self.model.predict(image, verbose=0)
            self.features[name] = feature
            print("> Image Identifier > ", name, "count >", count)

    def load_captions(self, caption_file):
        print('Loading Captions from dir ', caption_file)
        file = open(caption_file, 'r')
        text = file.read()
        file.close()

        for line in text.split('\n'):
            tokens = line.split()
            if len(line) < 2:
                continue
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            image_desc = ' '.join(image_desc)
            if image_id not in self.description:
                self.description[image_id] = list()
            self.description[image_id].append(image_desc)

    def cleaning_captions(self, clan_caption):
        if clan_caption:
            print('Cleaning Caption unwanted characters like single work and punctuations')
            table = str.maketrans('', '', string.punctuation)
            for key, desc_list in self.description.items():
                for i in range(len(desc_list)):
                    desc = desc_list[i]
                    desc = desc.split()
                    desc = [word.lower() for word in desc]
                    desc = [w.translate(table) for w in desc]
                    desc = [word for word in desc if len(word) > 1]
                    desc = [word for word in desc if word.isalpha()]
                    desc_list[i] = ' '.join(desc)

    def vocabulary(self):
        self.vocab = set()
        for key in self.description.keys():
            [self.vocab.update(d.split()) for d in self.description[key]]

    def enumurate_vocab_char_map(self):
        self.chars = set()
        for key in self.description.keys():
            for d in self.description[key]:
                temp_char = set(d)
                self.chars.update(temp_char)

    def save_feature_and_description(self):
        pickle.dump(self.description, open('description.pkl', 'wb'))
        pickle.dump(self.features, open('features.pkl', 'wb'))

    def load_identifiers(self, filename):
        # used to load the training and testing files and prepare the list of image identifiers
        file = open(filename, 'r')
        txt = file.read()
        file.close()

        dataset = list()
        for line in txt.split('\n'):
            if len(line) < 1:
                continue
            identifier = line.split('.')[0]
            dataset.append(identifier)

        return dataset

    def load_features(self, identifiers):
        features = {k: self.features[k] for k in identifiers}
        return features

    def load_description(self, identifiers):
        descriptions = dict()
        for image_id in identifiers:
            lst_desc = self.description[image_id]
            descriptions[image_id] = lst_desc
        return self.add_start_end_token_to_desc(descriptions)

    def add_start_end_token_to_desc(self, description):
        descriptions = dict()
        for image_id, desc_list in description.items():
            if image_id not in descriptions:
                descriptions[image_id] = list()
            for des in desc_list:
                tokens = des.split()
                desc = 'startseq ' + ' '.join(tokens) + ' endseq'
                descriptions[image_id].append(desc)
        return descriptions

    def to_lines(self, description):
        all_desc = list()
        for key in description.keys():
            [all_desc.append(d) for d in description[key]]
        return all_desc

    def create_tokens(self, description):
        tokenizer = Tokenizer()
        all_desc = self.to_lines(description)
        tokenizer.fit_on_texts(all_desc)
        tokenizer = tokenizer
        vocab_size = len(tokenizer.word_index) + 1

        # Length of Description with most words
        max_desc_length = max(len(line.split()) for line in all_desc)

        print('Vocabulary Size: %d' % vocab_size)
        print('Max Description Length: %d' % max_desc_length)
        return tokenizer, vocab_size, max_desc_length

    def create_sequence(self, description, tokenizer, max_desc_length, vocab_size, features):
        print('creating sequence ')
        X1, X2, Y = list(), list(), list()
        for key, desc_list in description.items():
            try:
                f = features[key]
                for desc in desc_list:
                    seq = tokenizer.texts_to_sequences([desc])[0]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_desc_length)[0]
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        X1.append(f[0])
                        X2.append(in_seq)
                        Y.append(out_seq)
            except:
                continue

        return array(X1), array(X2), array(Y)

    def define_caption_model(self, max_desc_length, vocab_size):
        # Feature model
        input_image_feature = Input(shape=(4096,))
        fe1 = Dropout(0.5)(input_image_feature)
        fe2 = Dense(256, activation='relu')(fe1)

        # Sequence Model
        input_sequence = Input(shape=(max_desc_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(input_sequence)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        # Decoder
        decoder1 = add([fe2,se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        output = Dense(vocab_size, activation='softmax')(decoder2)

        self.model_new = Model(inputs=[input_image_feature,input_sequence], output=output)
        self.model_new.compile(loss='categorical_crossentropy', optimizer='adam')

        print(self.model_new.summary())
        plot_model(self.model_new, to_file='model.png', show_shapes=True)

    def checkpoint_prepare(self):
        filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        self.checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    def model_fit(self, X1, X2, Y, X1_V,X2_V, Y_V):
        # fit model
        self.model_new.fit([X1, X2], Y, batch_size=50, epochs=5, verbose=1, callbacks=[self.checkpoint],
                  validation_data=([X1_V, X2_V], Y_V))

    def evaluate(self,X1, X2, Y):
        self.model_new.evaluate([X1, X2], Y, batch_size=10, verbose=1)