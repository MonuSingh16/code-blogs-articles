## Train Validation and Test Set

As we know, we begin by splitting data into :

1. Train
2. Validation
3. Test

At this point, remember that the test data is not available or does not even exit. So we begin with train data,
here we analyze it, transform it, determine features, and fit a model.

After modelling, we measure performance of model on validation set, which is unseen data. Based on the model performance on validation set, we improve our model. Hence we iteratively build our model.

Until, we reach a point where the model starts overfitting the validation set. So, we will merge this with train set and generate a new split of train and validation

Now, if we are happy with model performance we can evaluate on test data

However, if model is underperforming on test set, then we can go back to modelling stage. but here is important things
- Do not use the same test data again

As the test set is exposed now, and any previous evaluation might influence the future evaluation on that specific test set.
That's why you must use the a specific test set only once and generate entirely new splut after merging the train, validation and test set.

