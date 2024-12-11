files1=list(["data/uj_raw.txt","data/uj_raw2.txt","data/uj_raw3.txt","data/uj_raw4.txt","data/uj_raw5.txt","data/uj_raw+.txt",])
files2=list(["data/nieuj_raw.txt","data/nieuj_raw2.txt","data/nieuj_raw3.txt","data/nieuj_raw4.txt","data/nieuj_raw5.txt"])

with open('data/uj.csv','w') as f:
    #f.write("\"labels\",\"text\"\n")
    for file in files1:
        with open(file,'r') as g:
            data=g.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\"1\",\""+line+"\""
                #print(line)
                f.write(line+'\n')
    for file in files2:
        with open(file,'r') as h:
            data=h.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\"0\",\""+line+"\""
                #print(line)
                f.write(line+'\n')

import random
with open('data/uj.csv','r') as f:
    data=f.readlines()
    random.shuffle(data)
    with open('data/uj.csv','w') as g:
        g.write("\"labels\",\"text\"\n")
        for line in data:
            g.write(line)
