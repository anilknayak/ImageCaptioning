# Image Capitioning

Download model and pickle file from following folder [it has two folders 'model' and 'pkl']
https://drive.google.com/open?id=1gLhixTjhBkpeZkJ6DsYKzMdyCVkZUrtF

Send a download request from the below link for Flickr_8k_dataset
https://illinois.edu/fb/sec/1713398

You will be receiving an email to download the dataset. There are two zip file
Flickr8k_Dataset.zip [Images] place all the images into data/images folder
Flickr8k_text.zip [captions] place all the captions into data/caption folder

After you download all the required files, your directory structure will look like
-- ImageCaptioning
  -- data
    -- images
    -- caption
  -- pkl
    - details.pkl
    - features.pkl
    - tokenizer.pkl
    - description.pkl
  -- model
    - model-ep002-loss3.670-val_loss3.849.h5
    - model-ep005-loss3.226-val_loss3.783.h5
  -- ipython
    - ImageCaptioning.ipynb
    - model.png
  - captioning.py
  - gui.py
  - prepare.py
  - test_images.py
