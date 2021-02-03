from pathlib import Path
import random
import pandas as pd
import glob

# this file show how to randomize 1000 tweet ids per day across March to Oct
data_dirs = ["/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-03",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-04",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-05",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-06",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-07",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-08",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-09",
             "/Volumes/cdl_Drive/CMPT825_proj/data/USA/2020-10"]
    
if 'USA' in data_dirs[0]:
    for data_dir in data_dirs:
        print("[*] Generating ids for " + data_dir)
        p = Path(data_dir)
        all_files = glob.glob(data_dir + '/*.csv')
        output = open("/Volumes/cdl_Drive/CMPT825_proj/ids/USA/" + p.parts[-1] + '.txt', 'w')

        for fp in all_files:
            with open(fp, "rt", encoding='ascii') as f:
                for i, l in enumerate(f):
                    pass
                if i + 1 > 1000:
                    skip = sorted(random.sample(range(i + 1), i + 1 - 1000))
                    df = pd.read_csv(fp, skiprows=skip)
                else:
                    df = pd.read_csv(fp)
                for row in df.values:
                    output.write(str(row[0]) + '\n')
        output.close()
        print("[!] Done.")

else:
    for data_dir in data_dirs:
        p = Path(data_dir)
        all_files = glob.glob(data_dir + '/*.csv')
        output = open("/Volumes/cdl_Drive/CMPT825_proj/ids/CA/" + p.parts[-1] + '.txt', 'w')

        for fp in all_files:
            with open(fp, "rt", encoding='ascii') as f:
                for row in f:
                    output.write(row)
        output.close()

# CODE FOR PREVIOUS VERSION
# def random_id(file_path):
#     print("[*] Generating ids for " + file_path.parts[-1])
#     s = 1000  # desired sample size
#
#     with open(file_path) as f:
#         for i, l in enumerate(f):
#             pass
#     if i+1 > 1000:
#         skip = sorted(random.sample(range(i+1), i+1 - s))
#         df = pd.read_csv(file_path, skiprows=skip)
#     else:
#         df = pd.read_csv(file_path)
#     output = "/Volumes/cdl_Drive/CMPT825_proj/ids/USA/" + file_path.parts[-2] + '.txt'
#     with open(output, "w") as f:
#         for row in df.values:
#             f.write(str(row[0]) + '\n')
#     print("[!] Done.")
#

if __name__ == "__main__":
    main()
