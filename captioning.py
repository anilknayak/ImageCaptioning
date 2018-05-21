#-------------------------------------------------------------------
# @author 
# @copyright (C) 2018, 
# @doc
#
# @end
# Created : 19. May 2018 8:09 PM
#-------------------------------------------------------------------

import prepare
import pickle
class Captioning():
    def __init__(self,  sample=False,
                        model_name='VGG16',
                        image_dir='data/images',
                        caption_file='data/caption/Flickr8k.token.txt',
                        clean_caption=False,
                        load_pickle=False):
        self.prepare = prepare.Prepare(sample,model_name,image_dir,caption_file,clean_caption,load_pickle)
        training_dataset = self.prepare.load_identifiers('data/caption/Flickr_8k.trainImages.txt')
        testing_dataset = self.prepare.load_identifiers('data/caption/Flickr_8k.testImages.txt')
        validation_dataset = self.prepare.load_identifiers('data/caption/Flickr_8k.devImages.txt')

        train_feature = self.prepare.load_features(training_dataset)
        test_feature = self.prepare.load_features(testing_dataset)
        validation_feature = self.prepare.load_features(validation_dataset)

        train_description = self.prepare.load_description(training_dataset)
        test_description = self.prepare.load_description(testing_dataset)
        validation_description = self.prepare.load_description(validation_dataset)

        tokenizer, vocab_size, max_length = self.prepare.create_tokens(self.prepare.description)
        _, _, m1 = self.prepare.create_tokens(train_description)
        _, _, m2 = self.prepare.create_tokens(test_description)
        _, _, m3 = self.prepare.create_tokens(validation_description)
        max_length = max([max_length, m1, m2, m3])

        detail = dict()
        detail['vocab_size'] = vocab_size
        detail['max_length'] = max_length
        pickle.dump(tokenizer, open('pkl/tokenizer.pkl', 'wb'))
        pickle.dump(detail, open('pkl/detail.pkl', 'wb'))

        X1train, X2train, ytrain = self.prepare.create_sequence(train_description, tokenizer, max_length, vocab_size, train_feature)
        X1test, X2test, ytest = self.prepare.create_sequence(test_description, tokenizer, max_length,vocab_size, test_feature)
        X1validation, X2validation, yvalidation = self.prepare.create_sequence(validation_description, tokenizer, max_length,vocab_size, validation_feature)

        self.prepare.define_caption_model(max_length, vocab_size)
        self.prepare.checkpoint_prepare()
        self.prepare.model_fit(X1train, X2train, ytrain,X1validation, X2validation, yvalidation)


if __name__ == '__main__':
    Captioning()