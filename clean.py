# encoding=utf-8

import jieba

FILE_NAMES = ['pass.txt', 'fail.txt']
stop = [line.strip() for line in open('data/stop_words.txt', encoding="utf-8").readlines() ]

for filename in FILE_NAMES:
    words_list = [line.strip() for line in open('data/' + filename, encoding="utf-8").readlines()]
    dealed = []
    for line in words_list:
        words = jieba.cut(line)
        filtered = []
        for word in words:
            if word not in stop:
                filtered.append(word)
        dealed.append(' '.join(filtered))
    with open('clean/' + filename, 'a') as f:
        for line in dealed:
            f.write(line + '\n')
