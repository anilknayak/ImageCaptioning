# Image Capitioning

## Download pre trained model and pickle files
Download model and pickle file from following folder [it has two folders 'model' and 'pkl']
https://drive.google.com/open?id=1gLhixTjhBkpeZkJ6DsYKzMdyCVkZUrtF

## Download Flickr8k dataset
Send a request in the below link to download Flickr_8k_dataset
https://illinois.edu/fb/sec/1713398

You will be receiving an email to download the dataset. There are two zip file
1. Flickr8k_Dataset.zip [Images] place all the images into data/images folder
2. Flickr8k_text.zip [captions] place all the captions into data/caption folder

## Project Directory Structure
After you download all the required files, your directory structure will look like

    .
    ├── ImageCaptioning
        ├── data                    # data directory
        │   ├── images              # All the images from flickr8k dataset
        │   └── caption             # captions from flickr8k dataset
        ├── pkl                     # Pickle Files
        │   ├── details.pkl            # Details pickle has max description length
        │   └── features.pkl           # all image feature embedding
        │   └── tokenizer.pkl          # tokenizer for description
        │   └── description.pkl        # captions for each image
        └── model
        │   ├── model-ep002-loss3.670-val_loss3.849.h5            # model saved after epoch 2
        │   └── model-ep005-loss3.226-val_loss3.783.h5            # model saved after epoch 5
        └── ipython
        │   ├── ImageCaptioning.ipynb           # ipython notebook
        │   └── model.png                       # network model diagram
        ├── captioning.py           # training module
        ├── gui.py                  # gui module
        ├── prepare.py              # helper module
        └── test_images.py          # testing module


## How to run
1.  First Run the captioning.py for training
2.  Run test_images.py and provide a image path to test images
3.  Run gui.py for real work testing

## GUI 

(https://github.com/anilknayak/ImageCaptioning/blob/master/ipython/gui.png)
