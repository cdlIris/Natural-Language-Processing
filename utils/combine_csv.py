import csv
from pathlib import Path
import glob
import preprocessor as p
from nltk.corpus import stopwords
import re
from utils.prerpocess import preprocess_tweet

data_dirs = ["/Users/danlinchen/Documents/CMPT825/project/data_label/CA"]


for data_dir in data_dirs:
    all_files = glob.glob(data_dir + '/*.csv')

    csvFile_x = open('whole_text.csv', 'a')
    csvWriter_x = csv.writer(csvFile_x)

    csvFile_y = open('whole_label.csv', 'a')
    csvWriter_y = csv.writer(csvFile_y)

    for fp in all_files:
        print("[*] combining ", fp)
        with open(fp, 'r') as f:
            next(f)
            csvReader = csv.reader(f)
            for row in csvReader:
                text = preprocess_tweet(row[1])
                label = row[2]

                csvWriter_x.writerow([text])
                csvWriter_y.writerow([label])

    csvFile_x.close()
    csvFile_y.close()