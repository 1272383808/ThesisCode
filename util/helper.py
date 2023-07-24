import random


def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset0 = int(n_total * ratio[0])
    offset1 = int(n_total * ratio[1])
    offset2 = int(n_total * ratio[2])

    if n_total == 0: # 列表为空的情况
        return []

    if offset0 + offset1 + offset2 > n_total: # 错误切分条件
        print("错误切分比例:因为:", ratio[0], "+", ratio[1], "+", ratio[2], "=", ratio[0] + ratio[1] + ratio[2], ">1")
        return 0

    if offset0 + offset1 + offset2 <= n_total:# 切分
        random.shuffle(full_list)
    sublist_1 = full_list[:offset0]
    sublist_2 = full_list[offset0:offset0 + offset1]
    sublist_3 = full_list[offset0 + offset1:]
    return sublist_1, sublist_2, sublist_3