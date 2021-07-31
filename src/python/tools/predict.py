from pathlib import Path
import argparse
import numpy as np
import torch


def get_emb_dict(mapping_file, embs_file):
    mapping = np.loadtxt(mapping_file, dtype=str, delimiter='\t')
    mapping_dict = dict(mapping)
    mapping_r_dict = dict(zip(mapping[:,1], mapping[:,0]))

    num = len(mapping_dict)
    embs = np.fromfile(embs_file, np.float32).reshape((num, -1))

    return mapping_dict, mapping_r_dict, embs


def perform_link_prediction(data_dir, dataset_dir, node, rel, rel_type, k):
    node_mapping_file = Path(dataset_dir) / Path("node_mapping.txt")
    rel_mapping_file = Path(dataset_dir) / Path("rel_mapping.txt")
    node_embs_file = Path(data_dir) / Path("marius/embeddings/embeddings.bin")
    lhs_embs_file = Path(data_dir) / Path("marius/relations/src_relations.bin")
    rhs_embs_file = Path(data_dir) / Path("marius/relations/dst_relations.bin")

    node_mapping_dict, node_mapping_r_dict, node_embs = get_emb_dict(node_mapping_file, node_embs_file)

    if rel_type == 'lhs':
        rel_mapping_dict, rel_mapping_r_dict, rel_embs = get_emb_dict(rel_mapping_file, lhs_embs_file)
    else:
        rel_mapping_dict, rel_mapping_r_dict, rel_embs = get_emb_dict(rel_mapping_file, rhs_embs_file)

    node_id = int(node_mapping_dict.get(node))
    rel_id = int(rel_mapping_dict.get(rel))
    ctx_emb = node_embs[node_id] * rel_embs[rel_id]
    all_scores = torch.matmul(torch.tensor(node_embs), torch.tensor(ctx_emb))
    val, idx = all_scores.topk(k, largest = True)
    top_nodes = [node_mapping_r_dict.get(str(i.item())) for i in idx]

    return top_nodes


def print_nodes(nodes, k):
    print("Top " + k + " best matched nodes: ")
    print(nodes)


def set_args():
    parser = argparse.ArgumentParser(
        description='Perform link prediction',
        prog='predict'
    )
    parser.add_argument('trained_embeddings_directory',
                        metavar='trained_embeddings_directory',
                        type=str,
                        help='Directory containing trained embeddings')
    parser.add_argument('dataset_directory',
                        metavar='dataset_directory',
                        type=str,
                        help='Directory containing the dataset for training')
    parser.add_argument('k',
                        metavar='k',
                        type=str,
                        help='Number of predicted nodes to output')
    parser.add_argument('--src', '-s',
                        metavar='src',
                        type=str,
                        help='Source node, the original ID of a certain node')
    parser.add_argument('--dst', '-d',
                        metavar='dst',
                        type=str,
                        help='Destination node, the original ID of a certain node')
    parser.add_argument('--rel', '-r',  # what if there is only one relation?
                        metavar='rel',
                        type=str,
                        help='Relation, the original ID of a certain relation')
    parser.add_argument('--rel_type', '-rt',
                        default='lhs',
                        choices=['lhs', 'rhs'],
                        metavar='rel_type',
                        help='The direction of a certain relation')

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()
    emb_dir = args.trained_embeddings_directory
    dataset_dir = args.dataset_directory

    if (args.dst is None) and (not args.src is None):
        node = args.src
    elif (args.src is None) and (not args.dst is None):
        node = args.dst
    else:
        raise RunTimeError("Incorrect source node or destination node.")

    top_k_nodes = perform_link_prediction(emb_dir, dataset_dir, node, args.rel, args.rel_type, int(args.k))
    print_nodes(top_k_nodes, args.k)

if __name__=='__main__':
    main()