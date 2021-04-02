import random
from collections import OrderedDict
from hilbertcurve.hilbertcurve import HilbertCurve

import numpy as np


def flatten(tmp):
    # Converts a list of tuples [(a, b), (c, d)] to a list of values [a, b, c, d]
    ordering = []
    for t in tmp:
        ordering.append(t[0])
        ordering.append(t[1])
    return ordering


def evaluate_ordering(cache, ordering):
    hit = 0
    miss = 0
    for o in ordering:
        i = o[0]
        j = o[1]
        if cache.get(i) == -1:
            miss += 1
            cache.put(i, 0)
        else:
            hit += 1

        if cache.get(j) == -1:
            miss += 1
            cache.put(j, 0)
        else:
            hit += 1

    return hit, miss

def evaluate_ordering_ret_misses(cache, ordering):
    hit = 0
    miss = 0

    misses = []
    for o in ordering:
        i = o[0]
        j = o[1]
        if cache.get(i) == -1:
            miss += 1
            cache.put(i, 0)
            misses.append((i, j))
        else:
            hit += 1

        if cache.get(j) == -1:
            miss += 1
            cache.put(j, 0)
            misses.append((i, j))
        else:
            hit += 1

    return hit, miss, misses


def get_sequential_ordering(num_blocks):
    ordering = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            ordering.append((i, j))
    return ordering


def get_snake_ordering(num_blocks):
    ordering = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            idx = j
            if (i % 2 == 1):
                idx = (num_blocks) - j - 1

            ordering.append((i, idx))
    return ordering


def get_snake_symmetric_ordering(num_blocks):
    ordering = []
    for i in range(num_blocks):
        for j in range(i, num_blocks):
            idx = j

            if (i % 2 == 1):
                idx = (num_blocks) - (j - i) - 1

            if (i == j == num_blocks - 1):
                idx = j

            ordering.append((i, idx))
            if (i != idx):
                ordering.append((idx, i))

    return ordering


def get_random_ordering(num_blocks):
    ordering = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            ordering.append((i, j))

    random.shuffle(ordering)
    return ordering


def random_ordering_symmetric(num_blocks):
    tmp = get_random_ordering(num_blocks)
    ordering_set = OrderedDict()
    for ii in range((len(tmp))):
        c = tmp[ii]
        ordering_set[(c[0], c[1])] = None
        ordering_set[(c[1], c[0])] = None
        ordering_set[(c[0], c[0])] = None
        ordering_set[(c[1], c[1])] = None

    ordering = list(ordering_set.keys())
    return ordering


def hilbert_ordering(n):
    hilbert_curve = HilbertCurve(n, 2)
    coords = [hilbert_curve.coordinates_from_distance(i) for i in range(n ** 2)]
    ordering = []


    for ii in range((len(coords))):
        c = coords[ii]
        ordering.append((c[0], c[1]))

    return ordering


def hilbert_ordering_symmetric(n):
    hilbert_curve = HilbertCurve(n, 2)
    coords = [hilbert_curve.coordinates_from_distance(i) for i in range(n ** 2)]
    ordering_set = OrderedDict()
    for ii in range((len(coords))):
        c = coords[ii]
        ordering_set[(c[0], c[1])] = None
        ordering_set[(c[1], c[0])] = None
        ordering_set[(c[0], c[0])] = None
        ordering_set[(c[1], c[1])] = None

    ordering = list(ordering_set.keys())
    return ordering


def eval_room(room, meet, met):
    ordering = []
    for r in room:
        for j in room:
            if meet[r][j] != 1:
                meet[r][j] = 1
                if r == j == 0:
                    ordering.append((r, j))
                if meet[j][r] != 1:
                    ordering.append((r, j))
                    ordering.append((j, r))
                    if (meet[r][r] != 1):
                        meet[r][r] = 1
                        ordering.append((r, r))
                    if (meet[j][j] != 1):
                        meet[j][j] = 1
                        ordering.append((j, j))
                    met += 1
    return meet, met, ordering


def greedy_ordering(n, c):
    room = list(range(c))
    out = list(set(list(range(n))) - set(room))
    meet = np.zeros((n, n))

    max_meet = (n ** 2 - n) / 2

    ordering = []
    i_left = n - c
    met = 0

    rev = False
    swaps = 0

    while (met < max_meet):
        for i in range(i_left):
            meet, met, ordr = eval_room(room, meet, met)
            r = room[c - 1]
            if rev:
                o = out[-1 - i]
            else:
                o = out[i]
            room[c - 1] = o
            if rev:
                out[-1 - i] = r
            else:
                out[i] = r
            swaps += 1
            ordering = ordering + ordr

        meet, met, ordr = eval_room(room, meet, met)
        ordering = ordering + ordr
        room[0] = room[c - 1]

        rev = not rev

        if i_left >= c:
            for i in range(c - 1):
                if rev:
                    room[i + 1] = out[-1]
                    del out[-1]
                    swaps += 1
                else:
                    room[i + 1] = out[0]
                    del out[0]
                    swaps += 1
                i_left -= 1
                meet, met, ordr = eval_room(room, meet, met)
                ordering = ordering + ordr
        else:
            for i in range(i_left):
                if rev:
                    room[i + 1] = out[-1]
                    del out[-1]
                    swaps += 1
                else:
                    room[i + 1] = out[0]
                    del out[0]
                    swaps += 1
                i_left -= 1
                meet, met, ordr = eval_room(room, meet, met)
                ordering = ordering + ordr

    return ordering

# the greedy ordering has an analytic solution for number of misses
def get_misses(n, c):
    i = 0
    misses = n - c
    while ((n - c) - i * (c - 1) > 0):
        misses += (n - c) - i * (c - 1)
        i += 1
    return misses


# the greedy ordering has an analytic solution for number of misses
def get_lower_bound(n, c):
    num = (n * (n - 1) / 2) - (c * (c - 1) / 2)
    denom = c - 1
    return np.ceil(num / denom)

def get_opt():
    rooms = []
    n = 16
    c = 4
    with open("opt_rooms") as f:
        for line in f.readlines():
            rooms.append([int(i) for i in line.split(",")])

    ordering = []
    meet = np.zeros((n, n))
    met = 0

    max_meet = (n ** 2 - n) / 2

    ordering = []

    for room in rooms:
        meet, met, ordr = eval_room(room, meet, met)
        ordering = ordering + ordr

    return ordering