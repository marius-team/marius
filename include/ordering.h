//
// Created by Jason Mohoney on 7/17/20.
//

#ifndef MARIUS_ORDERING_H
#define MARIUS_ORDERING_H
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <batch.h>
#include <datatypes.h>
#include <config.h>

using std::pair;

vector<Batch *> applyOrdering(vector<Batch *> batches);

map<pair<int64_t, int64_t>, int64_t> getSequentialOrdering();

map<pair<int64_t, int64_t>, int64_t> getSequentialSymmetricOrdering();

map<pair<int64_t, int64_t>, int64_t> getRandomOrdering();

map<pair<int64_t, int64_t>, int64_t> getRandomSymmetricOrdering();

map<pair<int64_t, int64_t>, int64_t> getHilbertOrdering();

map<pair<int64_t, int64_t>, int64_t> getHilbertSymmetricOrdering();

map<pair<int64_t, int64_t>, int64_t> getEliminationOrdering();

map<pair<int64_t, int64_t>, int64_t> getCustomOrdering();

map<pair<int64_t, int64_t>, int64_t> makeOrderingSymmetric(map<pair<int64_t, int64_t>, int64_t> ordering_map);

int64_t xy2d(int64_t n, int64_t x, int64_t y);

void d2xy(int64_t n, int64_t d, int64_t *x, int64_t *y);

void rot(int64_t n, int64_t *x, int64_t *y, int64_t rx, int64_t ry);

#endif //MARIUS_ORDERING_H
