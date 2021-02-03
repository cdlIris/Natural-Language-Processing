
The analysis and BERT model are all run on Google Colab PRO. Since Google has installed most libraries in its own cloud environment, we didn't need to install a virtual environment to run our analysis. 

We did export all the packages we used in our analysis `requirement.txt`. However, we strongly recommend you run these two notebook in Google Colab Pro, since the Jupyter Notebooks are built on the cloud environment. 

we fetched our data from Twitter, and the regulation is that we can't share the data. Thus, we didn't share our full-dataset, which contains detailed tweet information in our GitLab, which means you might not be able to reproduce some of our results in these Notebook.¶

There are two Jupyter Notebooks here: one for Canada and one of the USA. Since we wanted to explore the USA and Canada data separately we created separate notebooks for each country. The pipeline and workflow are the same in each notebook, and the only difference is the dataset.

Both of the two notebooks covering the following information:
1.  Dataset Exploration
2.  Sentiment Change Analysis
3.  Most frequent terms Analysis
4.  Random Baseline
5.  Bert Model classification