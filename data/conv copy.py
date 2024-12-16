files1=list(["data/uj_raw.txt","data/uj_raw2.txt","data/uj_raw3.txt","data/uj_raw4.txt","data/uj_raw5.txt","data/uj_raw+.txt",])
files2=list(["data/train.csv","data/nieuj_raw.txt","data/nieuj_raw2.txt","data/nieuj_raw3.txt","data/nieuj_raw4.txt","data/nieuj_raw5.txt","data/nieuj_raw+.txt",])

import random

with open('data/uj.csv','w',encoding="utf8") as f:
    #f.write("\"labels\",\"text\"\n")
    for file in files1:
        with open(file,'r',encoding="utf8") as g:
            data=g.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line=line.replace(',','')
                line=line.replace('\"','')
                line2=line.split()
                random.shuffle(line2)
                line2=" ".join(line2)
                line="\"1\",\""+line+"\""
                line2="\"1\",\""+line2+"\""
                #print(line)
                f.write(line+'\n')
                f.write(line2+'\n')
    for file in files2:
        with open(file,'r',encoding="utf8") as h:
            data=h.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line=line.replace(',','')
                line=line.replace('\"','')
                line="\"0\",\""+line+"\""
                #print(line)
                f.write(line+'\n')

with open('data/uj.csv','r',encoding="utf8") as f:
    data=f.readlines()
    random.shuffle(data)
    with open('data/uj.csv','w',encoding="utf8") as g:
        g.write("\"labels\",\"text\"\n")
        for line in data:
            g.write(line)
