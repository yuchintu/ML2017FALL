import sys

filein = open(sys.argv[1])

#filein = open("words.txt")

d = dict()

for line in filein:
    words = line.split();
    for word in words:
        d[word] = d.get(word, 0) + 1

output = open("Q1.txt","w")
count = 0

for word, times in d.items():
    if(count < len(d) - 1):
        output.write(str(word) + " " + str(count) + " "  + str(times) + "\n")
        count += 1
    else:
        output.write(str(word) + " " + str(count) + " " + str(times))

