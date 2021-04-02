from collections import OrderedDict

import numpy as np

import buffer_simulator.orderings as order

import buffer_simulator.plotting

import argparse

class PerfectCache:
    # initialising capacity
    def __init__(self, capacity, ordering):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ordering = order.flatten(ordering)
        self.offset = 0
        self.size = 0
        i = 0
        while (len(self.cache.keys()) < self.capacity):
            self.put(self.ordering[i], 0)
            i += 1

    def find_least_useful(self):
        evict_k = None
        max_distance = 0
        for k in self.cache.keys():
            distance = 0
            for i in range(self.offset, len(self.ordering)):
                if (self.ordering[i] == int(k)):
                    break
                distance += 1
            if (distance > max_distance):
                evict_k = k
                max_distance = distance
        return evict_k
        # we return the value of the key

    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key):
        self.offset += 1
        if key not in self.cache:
            return -1
        else:
            return self.cache[key]

            # first, we add / update the key by conventional methods.

    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key, value):
        self.offset -= 1
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            lu = self.find_least_useful()
            del self.cache[lu]
        self.offset += 1


class LRUCache:
    # initialising capacity
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

        # we return the value of the key

    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key):
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

            # first, we add / update the key by conventional methods.

    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

        # RUNNER


def sequential(n, c, lru=False, output=True):
    ordering = order.get_sequential_ordering(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("Sequential (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def snake(n, c, lru=False, output=True):
    ordering = order.get_snake_ordering(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("Snake (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def snake_symmetric(n, c, lru=False, output=True):
    ordering = order.get_snake_symmetric_ordering(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("SnakeSymmetric (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def random_ordering(n, c, lru=False, output=True):
    ordering = order.get_random_ordering(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("Random (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def random_ordering_symmetric(n, c, lru=False, output=True):
    ordering = order.random_ordering_symmetric(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("RandomSymmetric (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def hilbert_ordering(n, c, lru=False, output=True):
    ordering = order.hilbert_ordering(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("Hilbert (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def hilbert_ordering_symmetric(n, c, lru=False, output=True):
    ordering = order.hilbert_ordering_symmetric(n)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)

    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("HilbertSymmetric (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses


def greedy_ordering(n, c, lru=False, output=True):
    ordering = order.greedy_ordering(n, c)

    if (lru):
        cache = LRUCache(c)
    else:
        cache = PerfectCache(c, ordering)
    hits, misses = order.evaluate_ordering(cache, ordering)
    if output:
        print("Greedy (N: %s, C: %s): Hits %s Misses %s Hit rate %s" % (n, c, hits, misses, hits / (hits + misses)))
    return hits, misses

