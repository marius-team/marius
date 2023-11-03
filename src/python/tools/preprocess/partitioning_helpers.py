import copy
import time
import sys
import random
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

                while counts[ii] > part_size > counts[jj]: # this doesn't need to be a loop, can do all at once
                    # move a random node from ii to jj
                    choice = np.random.choice(np.flatnonzero(parts == uniques_sorted_on_size[ii]))
                    parts[choice] = uniques_sorted_on_size[jj]
                    counts[ii] -= 1
                    counts[jj] += 1
            else:
                ii -= 1

        argsort = counts.argsort()
        uniques_sorted_on_size = uniques_sorted_on_size[argsort]
        counts = counts[argsort]

        # make only one partition smaller than part_size (this would be the last partition)
        for ii in range(1, counts.shape[0]):
            while counts[ii] < part_size:
                # move a random node from 0 to ii
                choice = np.random.choice(np.flatnonzero(parts == uniques_sorted_on_size[0]))
                parts[choice] = uniques_sorted_on_size[ii]
                counts[0] -= 1
                counts[ii] += 1

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
