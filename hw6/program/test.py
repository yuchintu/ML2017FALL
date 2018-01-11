import pickle
import sys

clusters = pickle.load(open('clusters_10.pkl','rb'))
print(len(clusters))
infile = open(sys.argv[1])
infile.readline()

test_case = []
for line in infile:
    line = line.split('\n')
    line = line[0].split(',')
    #print('1: '+str(clusters[int(line[1])]))
    #print('2: '+str(clusters[int(line[2])]))
    ans = 0
    if clusters[int(line[1])] == clusters[int(line[2])]:
        ans = 1

    #print(ans)
    test_case.append(ans)

infile.close()

result = open(sys.argv[2], 'w')
result.write('ID,Ans\n')
for i in range(len(test_case)):
    result.write(str(i) + ',' + str(int(test_case[i])) + '\n')






















