## Folder Description
- The `ieee` folder contains average sentiment analysis on the IEEE dataset
- The `lstm` folder contains lstm's tokenization code, model code and lstm evaluate code. And the saved models and logs along training.
- The `LSTM_RandomForest_Results.ipynb` is the visualization results of LSTM and Random forest. 
- The `randomForest` folder contains `RandomForest.ipynb` is the code for random forest models. 
- The `util` folder contains the preprocessing, labeling code, tweet hydrating code, tsv reader, and csv combiner code
- The `ids` folder contains the ids that we used to hydrate texts from tweepy api. 
- The `SentimentAnalysis_and_BERT` contains Sentiment Analysis and BERT model.

## SentimentAnalysis_and_BERT
This folder contains two Jupyter Notebook: one for Canada and one of the USA, covering the following information:
- Dataset Exploration
- Sentiment Change Analysis
- Most frequent terms Analysis
- Random Baseline
- Bert Model classification

## LSTM
Under the lstm folder, 
- `logs` contains the training and testing logs for CA and USA models.\
- `data` contains the traing and testing npy (lstm inputs) for CA and USA models. \
- `lstm_models` containts the saved models of lstm. \
- `lstm.py` contains the lstm model code. \
- `lstm_eval.py` contains the evalution code for LSTM. \
- `lstm_tokenization.py` contains the tokenization process from text. \

Use `python3 lstm.py` to train the lstm mdel, you could change the data dir, and parameter values inside it.\
Use `python3 lstm_eval.py` to evaluate the lstm model, make sure you have the correct data dir and model dir\