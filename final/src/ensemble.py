import re
import jieba
import numpy as np
import pdb
import pickle
import sys

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# set jieba dict
jieba.set_dictionary('./src/dict.txt.big')

# read our dictionary and the embedding layer

model = []
model1 = pickle.load(open('./src/model_1', 'rb'))
model.append(model1)
model2 = pickle.load(open('./src/model_2','rb'))
model.append(model2)
model3 = pickle.load(open('./src/model_3','rb'))
model.append(model3)

# produce the testing result
result_option = []
with open(sys.argv[1], 'r',encoding='utf8') as fr:
    # ignore the header
    next(fr)

    # read dialogue and answers
    for line_ori in fr:

        # read dialogue (question)
        line = line_ori
        line = line.strip().split(',')[1].replace("\t", " ")
        line = re.split(':| |A|B|C', line)
        sens = list(filter(None, line)) # remove all empty strings
        words = []
        for sen in sens:
                for word in list(jieba.cut(sen)):
                    words.append(word)
        temp_result_option = []
        for i in range(len(model)):
            feature_size = len(model[i]['等待'])
            # calculate dialogue vector
            emb_cnt = 0
            avg_dlg_emb = np.zeros((1,feature_size))
            for word in words:
                if word in model[i]:
                    avg_dlg_emb += model[i][word]
                    emb_cnt += 1
            if emb_cnt > 0:
                avg_dlg_emb /= emb_cnt

            # want to choose the best option
            idx = -1
            max_idx = -1
            max_sim = -10
            answers = line_ori.strip().split(',')[2]
            for ans in answers.strip().split(':')[1:]: # process one option for each iteration
                idx += 1

                # read options (answer)
                ans = ans.strip().replace("\t", " ")
                sens = re.split('A|B|C| ', ans)
                sens = list(filter(None, sens)) # remove all empty strings

                # calculate answer vectors
                # 在六個回答中，每個答句都取詞向量平均作為向量表示
                # 我們選出與dialogue句子向量表示cosine similarity最高的短句
                emb_cnt = 0
                avg_ans_emb = np.zeros((1,feature_size,))
                for sen in sens:
                    for word in jieba.cut(sen):
                        if word in model[i]:
                            if model[i][word].shape[0]==299:
                                pdb.set_trace()
                            avg_ans_emb += model[i][word]
                            emb_cnt += 1
                if emb_cnt > 0:
                    avg_ans_emb /= emb_cnt


                sim =cosine_similarity(avg_dlg_emb,avg_ans_emb)

                if sim > max_sim:
                    max_idx = idx
                    max_sim = sim

            # save answers to a temporary list
            temp_result_option.append(max_idx)

        if temp_result_option[1] == temp_result_option[2]:
            result_option.append(temp_result_option[1])
        else:
            result_option.append(temp_result_option[0])

# output our answer to file
with open(sys.argv[2], 'w') as fw:
    print('id,ans', file=fw)
    for idx, result in enumerate(result_option):
        print('%d,%d' % (idx+1, result), file=fw)
