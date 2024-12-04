

with open('data/uj_raw.txt','r') as f:
    with open('data/nieuj_raw.txt') as h:
        with open('data/uj.csv','w') as g:
            data=f.readlines()
            g.write("\"labels\",\"text\"\n")
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\"1\",\""+line+"\""
                #print(line)
                g.write(line+'\n')
            data=h.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\"0\",\""+line+"\""
                #print(line)
                g.write(line+'\n')
        