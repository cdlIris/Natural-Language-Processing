#!/usr/bin/env python3

# This script will walk through all the tweet id files and
# hydrate them with twarc (Need authroization of the api).
# partial citation from https://github.com/DocNow/twarc

import csv
from twarc import Twarc
from pathlib import Path


twarc = Twarc()
data_dirs = ["/Volumes/cdl_Drive/CMPT825_proj/ids/USA/"]


def main():
    for data_dir in data_dirs:
        for path in Path(data_dir).iterdir():
            if path.name.endswith('.txt'):
                hydrate(path)


def _reader_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    f = open(fname, 'rb')
    f_gen = _reader_generator(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)


def hydrate(id_file):
    print("[*] Hydrating ", id_file)
    print(id_file.parts)
    csvFile = open('data/' + id_file.parts[-1][:-4] + '.csv', 'a')
    csvWriter = csv.writer(csvFile)

    num_ids = raw_newline_count(id_file)

    for tweet in twarc.hydrate(id_file.open()):
        csvWriter.writerow([tweet["created_at"],
                            tweet["full_text"].encode('utf-8'),
                            tweet["user"]["location"]])
    csvFile.close()
    print("[!] Done.")
if __name__ == "__main__":
    main()
