import glob

dir_path = 'E:/AlanWattsMaterialSorted/text - Copy/'
read_files = glob.glob(dir_path + "*.txt")

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
