# encoding=utf-8

import jieba

stop = [line.strip().decode('utf-8') for line in open('stop_words.txt').readlines() ]

print(len(stop))

jieba.enable_paddle()
words = '你好啊a。'
a = jieba.cut(words)

for i in a:
    if i not in stop:
        print(i)
