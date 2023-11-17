import copy
import time
import sys
import random
import argparse
import numpy as np



def relabel_edges(edges, max_num_nodes, return_map=False):
    array_map = np.zeros(max_num_nodes, dtype=edges.dtype) - 1
    unique_nodes = np.unique(edges.flatten())
    array_map[unique_nodes] = np.arange(unique_nodes.shape[0], dtype=edges.dtype)
    mask = array_map == -1
    array_map[mask] = np.arange(unique_nodes.shape[0], max_num_nodes, dtype=edges.dtype)
    edges = array_map[edges.flatten()]
    edges = np.reshape(edges, (-1, 2))

    assert np.count_nonzero(edges.flatten() == -1) == 0

    if return_map:
        return edges, unique_nodes, array_map
    return edges, unique_nodes



def create_edge_buckets(edges, parts, cache_partitions=0, plot=True, partitioning_method=''):
    if True:
        summarize_parts(parts)

    num_partitions = np.unique(parts).shape[0]

    src_parts, dst_parts = parts[edges[:, 0]], parts[edges[:, -1]]
    sort_indices = np.lexsort((dst_parts, src_parts))
    edge_buckets = np.stack((src_parts[sort_indices], dst_parts[sort_indices]), axis=1)

    start_offsets = np.zeros([num_partitions, num_partitions], dtype=edges.dtype)
    src_splits = np.searchsorted(edge_buckets[:, 0], np.arange(num_partitions))

    for ii in range(num_partitions): # src partition index
        end_index = edges.shape[0] if ii == num_partitions - 1 else src_splits[ii + 1]
        dst_splits = np.searchsorted(edge_buckets[src_splits[ii]:end_index, -1], np.arange(num_partitions))
        dst_splits = dst_splits + src_splits[ii]
        start_offsets[ii, :] = dst_splits

    end_offsets = np.concatenate([np.reshape(start_offsets, [-1])[1:], edges.shape[0:1]], axis=0)
    end_offsets = np.reshape(end_offsets, [num_partitions, num_partitions])
    edge_bucket_sizes = end_offsets - start_offsets

    edges_diag = np.sum(np.diag(edge_bucket_sizes))
    edges_above = np.sum(np.triu(edge_bucket_sizes, 1))
    edges_below = np.sum(np.tril(edge_bucket_sizes, -1))
    if cache_partitions > 0:
        edges_diag_cache = np.sum(np.diag(edge_bucket_sizes)[cache_partitions:])
        edges_above_cache = edges_above
        edges_below_cache = edges_below
        for ii in range(cache_partitions):
            edges_above_cache -= np.sum(edge_bucket_sizes[ii, ii+1:])
            edges_below_cache -= np.sum(edge_bucket_sizes[ii+1:, ii])
    else:
        edges_diag_cache = edges_diag
        edges_above_cache = edges_above
        edges_below_cache = edges_below

    print("Assigned {} edges to edge_buckets".format(np.sum(edge_bucket_sizes)))
    print("Edges on diagonal: {}".format(edges_diag))
    print("Edges above diagonal: {}".format(edges_above))
    print("Edges below diagonal: {}".format(edges_below))
    print("Edges on diagonal ignoring {} cached partitions: {}".format(cache_partitions, edges_diag_cache))
    print("Edges above diagonal ignoring {} cached partitions: {}".format(cache_partitions, edges_above_cache))
    print("Edges below diagonal ignoring {} cached partitions: {}\n".format(cache_partitions, edges_below_cache))

    quality = 1 - (edges_above + edges_below)/edges.shape[0]
    quality_cache = 1 - (edges_above_cache + edges_below_cache)/edges.shape[0]

    # if plot:
    #     fig, ax = plt.subplots(1, 1)
    #     cs = ax.matshow(edge_bucket_sizes)
    #     plt.colorbar(cs)
    #     ax.set_title(partitioning_method + " partitioning with {} cached\nquality: {:.4f}, quality w/ cache: {:.4f}"
    #                  .format(cache_partitions, quality, quality_cache))
    #     plt.tight_layout()
    #     plt.show()
    #     # plt.savefig(partitioning_method+str(time.time)+'.pdf', bbox_inches='tight', format="pdf", transparent=False)

    return edge_bucket_sizes, quality, quality_cache



def random_partitioning(num_partitions, num_nodes):
    parts = np.random.randint(0, num_partitions, (num_nodes, ))
    return parts



def pymetis_partitioning(num_partitions, num_nodes, edges, recursive=True, duplicates=True):
    import pymetis

    # all num_nodes should be present in edges

    # adjacency_list = [np.array([1, 1, 1]),
    #                   np.array([0, 0, 0])]

    # undirected + duplicates
    edges_undirected = np.stack((edges[:, -1], edges[:, 0]), axis=1)
    edges_undirected = np.concatenate((edges, edges_undirected), axis=0)
    edges_undirected = edges_undirected[np.argsort(edges_undirected[:, 0])]
    adjacency_list = np.split(edges_undirected[:, 1], np.unique(edges_undirected[:, 0], return_index=True)[1][1:])

    # undirected + no duplicates
    if not duplicates:
        for ii in range(len(adjacency_list)):
            adjacency_list[ii] = np.unique(adjacency_list[ii])

    # n_cuts, membership = pymetis.part_graph(num_partitions, edges)
    n_cuts, membership = pymetis.part_graph(num_partitions, adjacency_list, recursive=recursive)
    membership = np.array(membership)

    print("pymetis: num nodes {}, edge cuts {}, assigned nodes {}\n".format(num_nodes, n_cuts, membership.shape))
    return membership



def summarize_parts(parts):
    uniques, counts = np.unique(parts, return_counts=True)
    print("unique partitions: ", uniques)
    print("partition counts: ", counts)



def add_missing_nodes(parts, num_nodes):
    print("Adding back missing nodes to {} total nodes".format(num_nodes))
    max_part = np.max(parts)
    new_parts = np.zeros(num_nodes - parts.shape[0], parts.dtype)

    idx = 0
    part = 0
    for ii in range(parts.shape[0], num_nodes):
        new_parts[idx] = part
        idx += 1
        part += 1

        if part > max_part:
            part = 0

    parts = np.concatenate((parts, new_parts), axis=0)

    return parts



def balance_parts(parts, part_size, edges):
    part_size = int(part_size)
    print("Balancing to: ", part_size)

    if edges is None:
        uniques, counts = np.unique(parts, return_counts=True) # uniques are sorted

        argsort = counts.argsort()
        uniques_sorted_on_size = uniques[argsort]
        counts = counts[argsort]

        # the merge
        ii = counts.shape[0] - 1
        while ii > 0:
            if counts[ii] > part_size:
                jj = 0
                while counts[jj] >= part_size and jj < ii:
                    jj += 1

                # move random nodes from ii to jj
                num_to_move = min(counts[ii]-part_size, part_size-counts[jj])
                if num_to_move > 0:
                    choices = np.random.choice(np.flatnonzero(parts == uniques_sorted_on_size[ii]), size=(num_to_move, ), replace=False)
                    parts[choices] = uniques_sorted_on_size[jj]
                    counts[ii] -= num_to_move
                    counts[jj] += num_to_move
            else:
                ii -= 1

        argsort = counts.argsort()
        uniques_sorted_on_size = uniques_sorted_on_size[argsort]
        counts = counts[argsort]

        # make only one partition smaller than part_size (this would be the last partition)
        for ii in range(1, counts.shape[0]):
            num_to_move = part_size - counts[ii]
            if num_to_move > 0:
                choices = np.random.choice(np.flatnonzero(parts == uniques_sorted_on_size[0]), size=(num_to_move, ), replace=False)
                parts[choices] = uniques_sorted_on_size[ii]
                counts[0] -= num_to_move
                counts[ii] += num_to_move

        argsort = counts.argsort()[::-1]
        uniques_sorted_on_size = uniques_sorted_on_size[argsort]
        counts = counts[argsort]

        tmp = uniques_sorted_on_size[-1]
        mask1 = parts == np.max(parts)
        mask2 = parts == tmp

        parts[mask2] = np.max(parts)
        parts[mask1] = tmp

    else:
        # TODO: instead of moving a random node, move the min score node?
        # Next level would be move the min score node to the small partition with the smallest penalty, but then you need to track a (num_nodes, num_parts)?
        pass

    return parts










# def tree_partitioning(num_partitions, num_nodes, edges, depth=0, root_number=0, global_parts=None, diff=0):
#
#     # parts = pymetis_partitioning(2, num_nodes, edges)
#     # parts = degree_partitioning(2, num_nodes, edges)
#     parts = custom_partitioning(2, num_nodes, edges)
#     # parts = union_find_partitioning(2, num_nodes, edges)
#     parts = parts + root_number
#     global_parts = copy.deepcopy(parts)
#
#     if depth < np.log2(num_partitions) - 1:
#         for x in [0, 1]:
#             edges_parts = np.stack((parts[edges[:, 0]], parts[edges[:, -1]]), axis=1)
#             mask = np.logical_and(edges_parts[:, 0] == edges_parts[:, 1], np.min(edges_parts, axis=1) == x + root_number)
#             remaining_edges = edges[mask]
#
#             remaining_edges, remaining_nodes_with_edges = relabel_edges(remaining_edges, num_nodes)
#             remaining_nodes = np.argwhere(parts == x + root_number)[:, 0]
#
#             num_remaining_nodes_with_edges = remaining_nodes_with_edges.shape[0]
#             num_remaining_nodes = remaining_nodes.shape[0]
#
#             new_parts = tree_partitioning(num_partitions, remaining_nodes_with_edges.shape[0], remaining_edges,
#                                           depth=depth+1, root_number=(root_number+x)*2,
#                                           diff=num_remaining_nodes-num_remaining_nodes_with_edges)
#
#             # # TODO: what to do with these remaining nodes
#             # parts[remaining_nodes] = random_partitioning(np.unique(new_parts).shape[0], num_remaining_nodes) + np.min(new_parts)
#             global_parts[remaining_nodes_with_edges] = new_parts
#
#     # print(np.unique(global_parts))
#     # print(global_parts)
#     return global_parts
#
# def custom_partitioning(num_partitions, num_nodes, edges, part_size=None, existing_parts=None, existing_scores=None):
#     if part_size is None:
#         part_size = np.ceil(num_nodes/num_partitions)
#
#     # sort the edges by src/dst or degree
#     # edges = edges[edges[:, 0].argsort()]
#
#     # degrees = degree_partitioning(num_partitions, num_nodes, edges, part_size=-1, return_degrees=True)
#     # edge_degrees = degrees[edges[:, 0]] + degrees[edges[:, 1]]
#     # edges = edges[edge_degrees.argsort()]
#     # edges = edges[edge_degrees.argsort()[::-1]]
#
#     chunk_size = 30000000#0 #27000 #6800000 #10000
#     num_chunks = int(np.ceil(edges.shape[0]/chunk_size))
#
#     if existing_parts is None:
#         parts = np.zeros(num_nodes, dtype=np.int32) - 1
#     else:
#         parts = existing_parts
#     if existing_scores is None:
#         part_scores = np.zeros((num_nodes, num_partitions), dtype=np.int32)
#         part_sizes = np.zeros(num_partitions, dtype=np.int32)
#     else:
#         assert existing_parts is not None
#         part_scores = existing_scores
#         # part_scores = np.zeros((num_nodes, num_partitions), dtype=np.int32)
#         part_sizes = np.zeros(num_partitions, dtype=np.int32)
#         for jj in range(num_partitions):
#             part_sizes[jj] += np.count_nonzero(parts == jj)
#
#     # part_scores = np.zeros((num_nodes, num_partitions), dtype=np.int32)
#     # part_0_scores = np.zeros(num_nodes, dtype=np.int32)
#     # part_1_scores = np.zeros(num_nodes, dtype=np.int32)
#
#     # edges_match = np.zeros(num_nodes, dtype=np.int32)
#     # edges_mismatch = np.zeros(num_nodes, dtype=np.int32)
#
#
#     for ii in range(num_chunks):
#         print("chunk: ", ii)
#         edge_chunk = edges[ii*chunk_size:(ii+1)*chunk_size]
#
#         remaining_edges, remaining_nodes_with_edges = relabel_edges(edge_chunk, num_nodes)
#
#         # new_parts = pymetis_partitioning(num_partitions, remaining_nodes_with_edges.shape[0], remaining_edges)
#         # parts[remaining_nodes_with_edges] = new_parts + i*2
#         # print(np.unique(parts[remaining_nodes_with_edges]))
#
#         if ii == 0 and existing_parts is None:
#             new_parts = pymetis_partitioning(num_partitions, remaining_nodes_with_edges.shape[0], remaining_edges)
#
#             parts[remaining_nodes_with_edges] = new_parts
#
#             for jj in range(num_partitions):
#                 part_sizes[jj] += np.count_nonzero(new_parts==jj)
#
#             remaining_edges_src = remaining_edges[remaining_edges[:, 0].argsort()]
#             remaining_edges_dst = remaining_edges[remaining_edges[:, -1].argsort()]
#
#             src_splits = np.searchsorted(remaining_edges_src[:, 0], np.arange(remaining_nodes_with_edges.shape[0]))
#             dst_splits = np.searchsorted(remaining_edges_dst[:, -1], np.arange(remaining_nodes_with_edges.shape[0]))
#
#             for jj, node in enumerate(remaining_nodes_with_edges):
#                 end_index = remaining_edges_src.shape[0] if jj == remaining_nodes_with_edges.shape[0] - 1 else src_splits[jj+1]
#                 src_edges = remaining_edges_src[src_splits[jj]:end_index]
#
#                 end_index = remaining_edges_dst.shape[0] if jj == remaining_nodes_with_edges.shape[0] - 1 else dst_splits[jj+1]
#                 dst_edges = remaining_edges_dst[dst_splits[jj]:end_index]
#
#
#                 # l_part_scores = np.zeros(num_partitions, dtype=np.int32)
#                 # for edge in src_edges:
#                 #     for kk in range(num_partitions):
#                 #         if parts[remaining_nodes_with_edges[edge[-1]]] == kk:
#                 #             l_part_scores += 1
#                 #             l_part_scores[kk] -= 1
#                 #             break
#                 #
#                 # for edge in dst_edges:
#                 #     for kk in range(num_partitions):
#                 #         if parts[remaining_nodes_with_edges[edge[0]]] == kk:
#                 #             l_part_scores += 1
#                 #             l_part_scores[kk] -= 1
#                 #             break
#
#
#                 l_part_scores_1 = np.zeros(num_partitions, dtype=np.int32)
#
#                 np.add.at(l_part_scores_1, parts[remaining_nodes_with_edges[src_edges[:, -1]]], -1)
#                 l_part_scores_1 += src_edges.shape[0]
#
#                 np.add.at(l_part_scores_1, parts[remaining_nodes_with_edges[dst_edges[:, 0]]], -1)
#                 l_part_scores_1 += dst_edges.shape[0]
#
#                 # assert np.array_equal(l_part_scores, l_part_scores_1), "ERROR 1"
#                 l_part_scores = l_part_scores_1
#
#                 # print(l_part_scores_t)
#                 # print()
#
#
#                 # part_0_scores[node] = part_0_score
#                 # part_1_scores[node] = part_1_score
#                 part_scores[node] = l_part_scores
#
#
#             # for edge in remaining_edges:
#             #     if parts[remaining_nodes_with_edges[edge[0]]] == parts[remaining_nodes_with_edges[edge[1]]]:
#             #         edges_match[remaining_nodes_with_edges[edge[0]]] += 1
#             #         edges_match[remaining_nodes_with_edges[edge[-1]]] += 1
#             #     else:
#             #         edges_mismatch[remaining_nodes_with_edges[edge[0]]] += 1
#             #         edges_mismatch[remaining_nodes_with_edges[edge[-1]]] += 1
#
#         else:
#             # new_parts_flip = new_parts + 1
#             # new_parts_flip[new_parts_flip == 2] = 0
#             #
#             # if np.count_nonzero(parts[remaining_nodes_with_edges] == new_parts) < np.count_nonzero(parts[remaining_nodes_with_edges] == new_parts_flip):
#             #     pass
#             # else:
#             #     new_parts = new_parts_flip
#             #
#             # for jj, node in enumerate(remaining_nodes_with_edges):
#             #     if parts[node] == -1:
#             #         parts[node] = new_parts[jj]
#             #
#             #         part_sizes[new_parts[jj]] += 1
#             #
#             #     if parts[node] == new_parts[jj]:
#             #         continue
#             #     else:
#             #         # how to decide whether to flip or not
#             #         part_0_score = 0
#             #         part_1_score = 0
#             #         for edge in remaining_edges:
#             #             if remaining_nodes_with_edges[edge[0]] == node:
#             #                 if parts[remaining_nodes_with_edges[edge[-1]]] == 0:
#             #                     part_1_score += 1
#             #                 if parts[remaining_nodes_with_edges[edge[-1]]] == 1:
#             #                     part_0_score += 1
#             #             elif remaining_nodes_with_edges[edge[-1]] == node:
#             #                 if parts[remaining_nodes_with_edges[edge[0]]] == 0:
#             #                     part_1_score += 1
#             #                 if parts[remaining_nodes_with_edges[edge[0]]] == 1:
#             #                     part_0_score += 1
#             #
#             #         if part_0_score < part_1_score:
#             #             parts[node] = 0
#             #             part_sizes[0] += 1
#             #         elif part_1_score > part_0_score:
#             #             parts[node] = 1
#             #             part_sizes[1] += 1
#             #         else:
#             #             choice = np.argmin(part_sizes)
#             #             # choice = random.randint(0, 1)
#             #             parts[node] = choice
#             #             part_sizes[choice] += 1
#
#
#
#
#
#             remaining_edges_src = remaining_edges[remaining_edges[:, 0].argsort()]
#             remaining_edges_dst = remaining_edges[remaining_edges[:, -1].argsort()]
#
#             src_splits = np.searchsorted(remaining_edges_src[:, 0], np.arange(remaining_nodes_with_edges.shape[0]))
#             dst_splits = np.searchsorted(remaining_edges_dst[:, -1], np.arange(remaining_nodes_with_edges.shape[0]))
#
#             for jj, node in enumerate(remaining_nodes_with_edges):
#                 end_index = remaining_edges_src.shape[0] if jj == remaining_nodes_with_edges.shape[0] - 1 else src_splits[jj + 1]
#                 src_edges = remaining_edges_src[src_splits[jj]:end_index]
#
#                 end_index = remaining_edges_dst.shape[0] if jj == remaining_nodes_with_edges.shape[0] - 1 else dst_splits[jj + 1]
#                 dst_edges = remaining_edges_dst[dst_splits[jj]:end_index]
#
#                 # part_0_score = 0
#                 # part_1_score = 0
#                 # for edge in src_edges:
#                 #     if parts[remaining_nodes_with_edges[edge[-1]]] == 0:
#                 #         part_1_score += 1
#                 #     if parts[remaining_nodes_with_edges[edge[-1]]] == 1:
#                 #         part_0_score += 1
#                 #
#                 # for edge in dst_edges:
#                 #     if parts[remaining_nodes_with_edges[edge[0]]] == 0:
#                 #         part_1_score += 1
#                 #     if parts[remaining_nodes_with_edges[edge[0]]] == 1:
#                 #         part_0_score += 1
#
#
#                 # l_part_scores = np.zeros(num_partitions, dtype=np.int32)
#                 # for edge in src_edges:
#                 #     for kk in range(num_partitions):
#                 #         if parts[remaining_nodes_with_edges[edge[-1]]] == kk:
#                 #             l_part_scores += 1
#                 #             l_part_scores[kk] -= 1
#                 #             break
#                 #
#                 # for edge in dst_edges:
#                 #     for kk in range(num_partitions):
#                 #         if parts[remaining_nodes_with_edges[edge[0]]] == kk:
#                 #             l_part_scores += 1
#                 #             l_part_scores[kk] -= 1
#                 #             break
#
#
#                 l_part_scores_1 = np.zeros(num_partitions, dtype=np.int32)
#
#                 temp = parts[remaining_nodes_with_edges[src_edges[:, -1]]]
#                 np.add.at(l_part_scores_1, temp[temp != -1], -1)
#                 l_part_scores_1 += np.count_nonzero(temp != -1)
#
#                 temp = parts[remaining_nodes_with_edges[dst_edges[:, 0]]]
#                 np.add.at(l_part_scores_1, temp[temp != -1], -1)
#                 l_part_scores_1 += np.count_nonzero(temp != -1)
#
#                 # assert np.array_equal(l_part_scores, l_part_scores_1), "ERROR 2"
#                 l_part_scores = l_part_scores_1
#
#
#                 if parts[node] == -1:
#                     argmin_part = np.argmin(l_part_scores)
#                     min_mask = l_part_scores == np.min(l_part_scores)
#                     if np.count_nonzero(min_mask) == 1 and part_sizes[argmin_part] < part_size: # and optional, TODO: do balancing better: pick min s.t. size < part_size
#                         parts[node] = argmin_part
#                         part_sizes[argmin_part] += 1
#                     else:
#                         temp_part_sizes = np.copy(part_sizes)
#                         temp_part_sizes[np.flatnonzero(l_part_scores != np.min(l_part_scores))] = num_nodes + 1
#                         choice = np.random.choice(np.flatnonzero(temp_part_sizes == np.min(temp_part_sizes)))
#                         parts[node] = choice
#                         part_sizes[choice] += 1
#
#                     # if part_0_score < part_1_score:
#                     #     parts[node] = 0
#                     #     part_sizes[0] += 1
#                     # elif part_1_score > part_0_score:
#                     #     parts[node] = 1
#                     #     part_sizes[1] += 1
#                     # else:
#                     #     choice = np.argmin(part_sizes)
#                     #     # choice = random.randint(0, 1)
#                     #     parts[node] = choice
#                     #     part_sizes[choice] += 1
#                 else:
#                     # part_scores_avg = l_part_scores
#                     # part_scores_avg = (1 - 1/(2*ii))*part_scores[node] + (1/(2*ii))*l_part_scores
#                     part_scores_avg = 0.5*part_scores[node] + 0.5*l_part_scores
#
#                     argmin_part = np.argmin(part_scores_avg)
#                     min_mask = part_scores_avg == np.min(part_scores_avg)
#                     if np.count_nonzero(min_mask) == 1:
#                         if parts[node] == argmin_part:
#                             pass
#                         else:
#                             if part_sizes[argmin_part] < part_size: # optional
#                                 temp = parts[node]
#                                 parts[node] = argmin_part
#                                 part_sizes[argmin_part] += 1
#                                 part_sizes[temp] -= 1
#                     else:
#                         temp_part_sizes = np.copy(part_sizes)
#                         temp_part_sizes[np.flatnonzero(part_scores_avg != np.min(part_scores_avg))] = num_nodes + 1
#                         choice = np.random.choice(np.flatnonzero(temp_part_sizes == np.min(temp_part_sizes)))
#                         if parts[node] == choice:
#                             pass
#                         else:
#                             if part_sizes[choice] < part_size: # optional
#                                 temp = parts[node]
#                                 parts[node] = choice
#                                 part_sizes[choice] += 1
#                                 part_sizes[temp] -= 1
#
#                     part_scores[node] = part_scores_avg
#
#
#                     # # part_0_score_avg = (1 - 1/(2*ii))*part_0_scores[node] + (1/(2*ii))*part_0_score
#                     # # part_1_score_avg = (1 - 1/(2*ii))*part_1_scores[node] + (1/(2*ii))*part_1_score
#                     # part_0_score_avg = 0.5*part_0_scores[node] + 0.5*part_0_score
#                     # part_1_score_avg = 0.5*part_1_scores[node] + 0.5*part_1_score
#                     #
#                     # if part_0_score_avg < part_1_score_avg:
#                     #     if parts[node] == 0:
#                     #         pass
#                     #     else:
#                     #         parts[node] = 0
#                     #         part_sizes[0] += 1
#                     #         part_sizes[1] -= 1
#                     # elif part_1_score_avg > part_0_score_avg:
#                     #     if parts[node] == 1:
#                     #         pass
#                     #     else:
#                     #         parts[node] = 1
#                     #         part_sizes[1] += 1
#                     #         part_sizes[0] -= 1
#                     # else:
#                     #     choice = np.argmin(part_sizes)
#                     #     # choice = random.randint(0, 1)
#                     #
#                     #     if parts[node] == choice:
#                     #         pass
#                     #     else:
#                     #         parts[node] = choice
#                     #         part_sizes[choice] += 1
#                     #         part_sizes[(choice+1)%2] -= 1
#                     #
#                     # part_0_scores[node] = part_0_score_avg
#                     # part_1_scores[node] = part_1_score_avg
#
#
#
#             # for edge in remaining_edges:
#             #     if parts[remaining_nodes_with_edges[edge[0]]] == parts[remaining_nodes_with_edges[edge[1]]]:
#             #         edges_match[remaining_nodes_with_edges[edge[0]]] += 1
#             #         edges_match[remaining_nodes_with_edges[edge[-1]]] += 1
#             #     else:
#             #         edges_mismatch[remaining_nodes_with_edges[edge[0]]] += 1
#             #         edges_mismatch[remaining_nodes_with_edges[edge[-1]]] += 1
#
#             # print(node, remaining_nodes_with_edges[jj])
#             # print(src_edges)
#             # print(dst_edges)
#             # exit()
#
#         # edge_bucket_sizes, q1, q2 = create_edge_buckets(remaining_edges, new_parts, 0)
#         # fig, ax = plt.subplots(1, 1)
#         # cs = ax.matshow(edge_bucket_sizes)
#         # plt.colorbar(cs)
#         # ax.set_title("Custom" + " partitioning with {} cached\nquality: {:.4f}, quality w/ cache: {:.4f}".format(0, q1, q2))
#         # plt.tight_layout()
#         # plt.show()
#
#     # least_matches = edges_match.argsort()
#     # most_mismatches = edges_mismatch.argsort()[::-1]
#     # print(least_matches)
#     # print(most_mismatches)
#     #
#     # for ii in range(15):
#     #     temp = parts[most_mismatches[ii]]
#     #     parts[most_mismatches[ii]] = parts[least_matches[ii]]
#     #     parts[least_matches[ii]] = temp
#
#
#
#     # exit()
#     return parts #, part_scores
#
#
#
#
#
#
#
#
#
#
# def check_partitioning(path="../Marius2P/datasets/ogb_wikikg90mv2_metis/", num_nodes=91230610, num_partitions=4096):
#     part_size = np.ceil(num_nodes/num_partitions)
#
#     train_edges = np.fromfile(path + "edges/train_edges.bin", dtype=np.int32)
#     train_edges = np.reshape(train_edges, (-1, 3))
#     train_edges = np.stack((train_edges[:, 0], train_edges[:, -1]), axis=1)
#
#     edge_bucket_sizes = np.genfromtxt(path + "edges/train_partition_offsets.txt", dtype=np.int32)
#     edge_bucket_ends = edge_bucket_sizes.cumsum()
#     edge_bucket_starts = edge_bucket_ends - edge_bucket_sizes
#
#     running_edge_bucket = 0
#     running_edge_bucket_count = 0
#
#     for ii in range(train_edges.shape[0]):
#         if ii % 1E7 == 0:
#             print(ii)
#
#         src_part = train_edges[ii, 0]//part_size
#         dst_part = train_edges[ii, -1]//part_size
#
#         edge_bucket_index = int(src_part*num_partitions + dst_part)
#
#         if edge_bucket_index != running_edge_bucket:
#             if not (running_edge_bucket_count == edge_bucket_ends[running_edge_bucket] - edge_bucket_starts[running_edge_bucket]):
#                 print("index: ", ii)
#                 print("edge: ", train_edges[ii])
#                 print("partitions: ", src_part, dst_part)
#                 print("running edge bucket index: ", running_edge_bucket)
#                 print("running edge bucket count: ", running_edge_bucket_count)
#                 print("Expected size: ", edge_bucket_ends[running_edge_bucket] - edge_bucket_starts[running_edge_bucket])
#                 raise Exception("TEST FAILED: Edge Bucket Size Mismatch")
#             running_edge_bucket = edge_bucket_index
#             running_edge_bucket_count = 1
#         else:
#             running_edge_bucket_count += 1
#
#         if not (edge_bucket_starts[edge_bucket_index] <= ii < edge_bucket_ends[edge_bucket_index]):
#             print("index: ", ii)
#             print("edge: ", train_edges[ii])
#             print("partitions: ", src_part, dst_part)
#             print("edge bucket index: ", edge_bucket_index)
#             print("edge bucket offsets: ", edge_bucket_starts[edge_bucket_index], edge_bucket_ends[edge_bucket_index+1])
#             raise Exception("TEST FAILED: Edges Out Of Order")
#
#     print("TEST PASSED")
#
#
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--path", required=True, help="path to dataset directory")
#     p.add_argument("--num_nodes", default=1, type=int)
#     p.add_argument("--num_partitions", default=2, type=int)
#
#     p = p.parse_args()
#
#     check_partitioning(p.path, p.num_nodes, p.num_partitions)