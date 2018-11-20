This experimentation framework aims to allow easy definition of new ML experiments

The module experiment, in experiment_setup, contains a chain of builders which allows to define an experiment in
a fluent API. Every step adds a wrapper to the experiment. The definition steps are the following:

1: Initiate the process by specifying the test/train split method. Options are kfold and single split.
2: Specify which learning algorithm to use. The algorithms are defined in the algorithms modules and contains the method
   used to optimize its hyperparameters (if needed) and the method to run an experiment.
3: Specify the validation method. Options are to reserve a split of the dataset for validation testing, to directly use
   the results of the train/test without validation (for the cases with no hyperparameters to optimize) and to use a
   distinct validation set.
4: Add a list of tokenizers. Those are defined in the vectorization module (Vectorizer.py and VectorizerSet.py) and will
   be used in the vectorization process instead of the built in sklearn tokenizer. The experiment will be repeated for
   every vectorizer. The user can also choose not to use a vectorizer if he intend to vectorize the data himself or if
   the data is not text-based and as such need no tokenization
5: Add output to console or to disk files.
6: Launch the experiment, passing in the data and targets.

The experiments themselves are defined in bdrv_experiments.py, carcomplaints_experiments.py and
 generalisation_experiment.py

 Experiment results are found in the /result folder and subfolders. The report only presents highlight, but all results
 can be found there.