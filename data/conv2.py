files1=list(["data/uj_val.txt"])
files2=list(["data/nieuj_val.txt"])

with open('data/uj2.csv','w') as f:
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
with open('data/uj2.csv','r') as f:
    data=f.readlines()
    random.shuffle(data)
    with open('data/uj2.csv','w') as g:
        g.write("\"labels\",\"text\"\n")
        for line in data:
            g.write(line)
