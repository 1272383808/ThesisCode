# import tokenization
# from util.get_tokens import *
# from util.mapping import *
#
# f = open('slicetxt/ArithmeticExpression/AE_dev.txt', 'r')
# slicelists = f.read().split("------------------------------")  # 按切片分割
# f.close()
#
# if slicelists[0] == '':  # 删除切片文件，前后的冗余信息
#     del slicelists[0]
# if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
#     del slicelists[-1]
#
# data = []
# index = 0
# for slicelist in slicelists:
#     if index == 0:
#         sentences = slicelist.split('\n')[1:]  # 以行为单位，对切片进行分割(先不要第一行的信息)
#     else:
#         sentences = slicelist.split('\n')[2:]  # 以行为单位，对切片进行分割(先不要第一行的信息)
#     if sentences[0] == '\r' or sentences[0] == '':  # 删除每一个切片前后无关的信息
#         del sentences[0]
#     if sentences == []:
#         continue
#     if sentences[-1] == '':
#         del sentences[-1]
#     if sentences[-1] == '\r':
#         del sentences[-1]
#
#     label = str(sentences[-1].strip())
#     slice_corpus = []
#     for sentence in sentences:  # 对每个切片的每一行进行单独处理
#         list_tokens = create_tokens(sentence)
#         slice_corpus.append(list_tokens)
#     if index % 100 == 0:
#         print("slicelist", index, ":", label)
#     sentencesAll = ''  # 记录每个切片（名字替换后）的所有token
#     slice_corpus, slice_func = mapping(slice_corpus)
#     for s in slice_corpus:
#         # print(s)
#         if sentencesAll == '':
#             sentencesAll = s
#         else:
#             sentencesAll = sentencesAll + ' ' + s
#     data.append(sentencesAll)
#     text_a = tokenization.convert_to_unicode(sentencesAll)
#     print(text_a)
#     index += 1
#
# print(len(data))
import numpy as np
import torch

a = torch.Tensor([[-0.1683, 0.1169],
        [-0.2445, -0.3385],
        [-0.2192, -0.3541],
        [-0.1699, -0.3324],
        [-0.2003, -0.3170],
        [-0.2339, -0.3778],
        [-0.2217, -0.3470],
        [-0.1710, -0.3188],
        [-0.2064, -0.3533],
        [-0.1895, -0.3201],
        [-0.2246, -0.3507],
        [-0.2117, -0.3494],
        [-0.1985, -0.3209],
        [-0.1863, -0.3291],
        [-0.2201, -0.3736],
        [-0.2259, -0.3554]])
print(np.argmax(a, axis=1))
