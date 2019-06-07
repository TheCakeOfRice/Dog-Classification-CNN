# Dog-Classification-CNN
Using a keras convolutional neural network to classify pictures of dogs.

The dataset used for this project may be downloaded at https://www.kaggle.com/jessicali9530/stanford-dogs-dataset.

Of this dataset, we use only three classes:
  - n02110958-pug (200 images)
  - n02110185-Siberian_husky (192 images)
  - n02099601-golden_retriever (150 images)
  
This is done wholly for computational considerations, since the model will be trained on a personal computer which cannot support GPU computations.

The results of the training and the performance on a validation set are graphed in acc.png and loss.png.  From these, it can be seen that the model begins to overfit the data past about 15 epochs, where the validation loss is minimized and accuracy tops out around 65%.  Though this accuracy is not great, it is still about twice as good as a random classifier (33%), and our complex model has learned quite a lot using a minimal amount of data.
