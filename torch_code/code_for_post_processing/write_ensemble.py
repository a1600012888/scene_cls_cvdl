#!/usr/bin/env mdl

import numpy as np
from test import write_result

test_info_path = '../data/test_info'
info_paths = [
    '../data/answer1.info',
    '../data/answer2.info',
    '../data/answer3.info',
    '../data/answer4.info',
]

answer_path = '../data/answer.info'

def read_top_3(info_path):

    dic = {}
    with open(info_path) as f:
        for line in f:
            #print(line)
            l_name = line[5:15]
            l_l = line[16:]

            #print(l_l)
            label = [int(x) for x in l_l.split(' ', 3)]
            #print(l_name, label)
            dic[l_name] = label
    return dic

def ensemble(dicts):

    ans = {}
    dic = dicts[0]

    for name in dic.keys():
        a = np.zeros(shape = (80), dtype = np.int8)

        for di in dicts:
            a[di[name]] += 1

        a = np.argsort(a)
        ans[name] = a[-3:]

    write_result(test_info_path, ans, answer_path)

    print('Writen done! in {}'.format(answer_path))


if __name__ == '__main__':

    dics = []
    for i_path in info_paths:
        dics.append(read_top_3(i_path))
    ensemble(dics)
# vim: ts=4 sw=4 sts=4 expandtab
