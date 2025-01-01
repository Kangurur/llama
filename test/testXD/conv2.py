
with open('testXD/data_raw.txt', 'r',encoding="utf8") as f:
    with open('testXD/data.csv', 'w',encoding="utf8") as f2:
        f2.write("\"text\"\n")
        lines=f.readlines()
        labels=lines[0].split(';')
        lines=lines[1:]
        for line in lines:
            line=line.split(';')
            line[-1]=line[-1][:-1]
            #print(line)
            #f2.write("\""+line[0]+" to "+line[1]+"\"\n")
            f2.write("\""+line[0]+" "+line[2]+"\"\n")
            f2.write("\""+line[1]+" "+line[2]+"\"\n")
            f2.write("\""+line[0]+" mieszka "+line[4]+"\"\n")
            f2.write("\""+line[1]+" mieszka "+line[4]+"\"\n")
            
        
        #print(labels)
        