
with open('testXD/data_raw.txt', 'r',encoding="utf8") as f:
    with open('testXD/data.csv', 'w',encoding="utf8") as f2:
        f2.write("\"question\";\"answer\"\n")
        lines=f.readlines()
        labels=lines[0].split(';')
        lines=lines[1:]
        for line in lines:
            line=line.split(';')
            line[-1]=line[-1][:-1]
            #print(line)
            f2.write("\"Pytanie: Kim jest "+line[0]+"? Odpowiedź: \";\""+line[0]+" to "+line[1]+", "+line[2]+"\"\n")
            f2.write("\"Pytanie: Kim jest "+line[1]+"? Odpowiedź: \";\""+line[1]+" to "+line[0]+", "+line[2]+"\"\n")
            f2.write("\"Pytanie: Gdzie mieszka "+line[0]+"? Odpowiedź: \";\""+line[0]+" mieszka "+line[3]+"\"\n")
            f2.write("\"Pytanie: Gdzie mieszka "+line[1]+"? Odpowiedź: \";\""+line[1]+" mieszka "+line[3]+"\"\n")
            f2.write("\"Pytanie: Powiedz coś jeszcze o "+line[0]+" Odpowiedź: \";\""+line[4]+"\"\n")
            f2.write("\"Pytanie: Powiedz coś jeszcze o "+line[1]+" Odpowiedź: \";\""+line[4]+"\"\n")
            f2.write("\"Pytanie: Opisz "+line[0]+" Odpowiedź: \";\""+line[0]+" "+line[2]+"\"\n")
            f2.write("\"Pytanie: Opisz "+line[1]+" Odpowiedź: \";\""+line[1]+" "+line[2]+"\"\n")
            
        
        #print(labels)
        