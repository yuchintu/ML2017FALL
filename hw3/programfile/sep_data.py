import pickle

label = []
pixel = []

with open('train.csv','r') as f:
    f.readline()
    for i, line in enumerate(f):
        data = line.split(',')
        label.append(data[0])
        pixel.append(data[1])
        
f.close()

train_pixel = open('train_pixel.pkl', 'wb')
pickle.dump(pixel[0:int(len(pixel) / 10 * 9)], train_pixel)
train_pixel.close()

train_label = open('train_label.pkl', 'wb')
pickle.dump(label[0:int(len(pixel) / 10 * 9)], train_label)
train_label.close()

test_with_ans_pixels = open('test_with_ans_pixels.pkl', 'wb')
pickle.dump(pixel[int(len(pixel) / 10 * 9):], test_with_ans_pixels)
test_with_ans_pixels.close()

test_with_ans_labels = open('test_with_ans_labels.pkl', 'wb')
pickle.dump(label[int(len(pixel) / 10 * 9):], test_with_ans_labels)
test_with_ans_labels.close()

'''
train = open('train_pixel.pkl', 'rb')
pixels = pickle.load(train)
train.close()
'''


































