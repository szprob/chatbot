import random


def make_wiki_question(q):
    r = random.random()
    if r < 0.1:
        res = "什么是" + q + "?"
    if r < 0.2:
        res = "什么是" + q + "？"
    if r < 0.3:
        res = q + "是什么"
    if r < 0.4:
        res = q + "是什么?"
    if r < 0.5:
        res = q + "是什么？"
    if r < 0.6:
        res = q + "是？"
    if r < 0.7:
        res = q + "是?"
    else:
        res = q
    return res


def split_list_by2(lst):
    result = []
    for i in range(2, len(lst) + 1, 2):
        result.append(lst[:i])
    return result
