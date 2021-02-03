import csv
"""
 This code is used to get the tweet ids corresponding to covid from https://zenodo.org/record/4192295#.X82YVy_r01J
 
 The dataset is full_dataset_clean.tsv. We uploaded the ids that we extracted under ids folder. 
"""

fs=","
table = str.maketrans('\t', fs)
fName = '/Volumes/cdl_Drive/CMPT825_proj/full_dataset_clean.tsv'
f = open(fName,'r')

try:
    line = f.readline()
    line = f.readline()
    while line:
        temp_row = str(line.translate(table))[:-1].split(',')
        tweet_id = temp_row[0]
        date = temp_row[1]
        lang = temp_row[3]
        region = temp_row[4]

        if lang == 'en' and region == 'CA' and date > '2020-03-04':
            print(temp_row)
            csvFile = open("/Volumes/cdl_Drive/CMPT825_proj/ids/CA/" + date + '.csv', 'a')
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow([tweet_id])
            csvFile.close()
        if lang == 'en' and region == 'US' and date > '2020-08-07':
            print(temp_row)
            csvFile = open("/Volumes/cdl_Drive/CMPT825_proj/ids/USA/" + date + '.csv', 'a')
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow([tweet_id])
            csvFile.close()

        line = f.readline()
except IOError:
    print("Could not read file: " + fName)
