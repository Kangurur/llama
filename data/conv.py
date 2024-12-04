

with open('data/uj_raw.txt','r') as f:
    with open('data/nieuj_raw.txt') as h:
        with open('data/uj.csv','w') as g:
            data=f.readlines()
            g.write("\"text\",\"label\"\n")
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\""+line+"\",\"1\""
                #print(line)
                g.write(line+'\n')
            data=h.readlines()
            for line in data:
                line=line[line.find(' ')+1:-3]
                line="\""+line+"\",\"0\""
                #print(line)
                g.write(line+'\n')
        