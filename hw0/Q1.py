import sys

filein = open(sys.argv[1])

#filein = open("words.txt")

d = dict()
l = []

for line in filein:
    words = line.split();
    for word in words:
        if(d.get(word, 0) == 0):
            l.append(word)
        d[word] = d.get(word, 0) + 1

output = open("Q1.txt","w")
count = 0

for word in l:
    if(count < len(l) - 1):
        output.write(str(word) + " " + str(count) + " "  + str(d.get(word)) + "\n")
        count += 1
    else:
        output.write(str(word) + " " + str(count) + " " + str(d.get(word)))

output.close()
