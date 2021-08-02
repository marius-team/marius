from pathlib import Path
import argparse
import numpy as np
import torch
import marius as m
import torch


def get_emb_dict(mapping_file, embs_file):
    mapping = np.loadtxt(mapping_file, dtype=str, delimiter='\t')
    mapping_dict = dict(mapping)
    mapping_r_dict = dict(zip(mapping[:,1], mapping[:,0]))

    num = len(mapping_dict)
    embs = np.fromfile(embs_file, np.float32).reshape((num, -1))

    return mapping_dict, mapping_r_dict, embs


def dismult_infer(node_emb, rel_emb, all_embs):
    if rel_emb is not None:
        ctx_emb = node_emb * rel_emb
    else:
        ctx_emb = node_emb
    all_scores = torch.matmul(torch.tensor(all_embs), torch.tensor(ctx_emb))

    return all_scores


def complex_infer(node_emb, rel_emb, all_embs):
    node_emb = torch.tensor(node_emb)
    rel_emb = torch.tensor(rel_emb)
    all_embs = torch.tensor(all_embs)

    if rel_emb is not None:
        dim = node_emb.size()[0]
        real_len = int(dim / 2)
        imag_len = int(dim - dim / 2)

        node_real_emb = node_emb.narrow(0, 0, int(real_len))
        node_imag_emb = node_emb.narrow(0, int(real_len), int(imag_len))
        rel_real_emb = rel_emb.narrow(0, 0, int(real_len))
        rel_imag_emb = rel_emb.narrow(0, int(real_len), int(imag_len))

        ctx_emb = torch.zeros_like(node_emb)
        ctx_emb[0:real_len] = (node_real_emb * rel_real_emb) - (node_imag_emb * rel_imag_emb)
        ctx_emb[real_len:] = (node_real_emb * rel_imag_emb) + (node_imag_emb * rel_real_emb)
    else:
        ctx_emb = node_emb
    
    all_scores = torch.matmul(all_embs, ctx_emb)

    return all_scores


def transe_infer(node_emb, rel_emb, all_embs):
    node_emb = torch.tensor(node_emb)
    rel_emb = torch.tensor(rel_emb)
    all_embs = torch.tensor(all_embs)

    if rel_emb is not None:
        ctx_emb = node_emb + rel_emb
    else:
        ctx_emb = node_emb

    ctx_emb = torch.reshape(ctx_emb, (1, -1))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    all_scores = cos(all_embs, ctx_emb)

    return all_scores


def perform_link_prediction(data_dir, dataset_dir, infer_list, k, decoder):
    node_mapping_file = Path(dataset_dir) / Path("node_mapping.txt")
    rel_mapping_file = Path(dataset_dir) / Path("rel_mapping.txt")
    node_embs_file = Path(data_dir) / Path("marius/embeddings/embeddings.bin")
    lhs_embs_file = Path(data_dir) / Path("marius/relations/src_relations.bin")
    rhs_embs_file = Path(data_dir) / Path("marius/relations/dst_relations.bin")

    node_mapping_dict, node_mapping_r_dict, node_embs = get_emb_dict(node_mapping_file, node_embs_file)
    top_nodes_list = []
    for i in infer_list:
        if i[1] != "":
            if i[2] == "" and i[0] != "":
                rel_mapping_dict, rel_mapping_r_dict, rel_embs = get_emb_dict(rel_mapping_file, lhs_embs_file)
                node_emb = node_embs[int(node_mapping_dict.get(i[0]))]
                rel_emb = rel_embs[int(rel_mapping_dict.get(i[1]))]
            elif i[2] != "" and i[0] == "":
                rel_mapping_dict, rel_mapping_r_dict, rel_embs = get_emb_dict(rel_mapping_file, rhs_embs_file)
                node_emb = node_embs[int(node_mapping_dict.get(i[2]))]
                rel_emb = rel_embs[int(rel_mapping_dict.get(i[1]))]
            else:
                raise RuntimeError("Incorrect inference format on line " + str(i))
        else:
            rel_emb = None
            if i[2] == "" and i[0] == "":
                raise RuntimeError("Incorrect inference format on line " + str(i))
            elif i[0] != "":
                node_emb = node_embs[int(node_mapping_dict.get(i[0]))]
            else:
                node_emb = node_embs[int(node_mapping_dict.get(i[2]))]
        
        if decoder == 'DisMult':
            all_scores = dismult_infer(node_emb, rel_emb, node_embs) 
        elif decoder == 'TransE':
            all_scores = transe_infer(node_emb, rel_emb, node_embs)
        elif decoder == 'ComplEx':
            all_scores = complex_infer(node_emb, rel_emb, node_embs)

        val, idx = all_scores.topk(k, largest=True)
        top_nodes = [node_mapping_r_dict.get(str(j.item())) for j in idx]
        top_nodes_list.append(top_nodes)

    return top_nodes_list


def print_nodes(nodes, k):
    print("Top " + k + " best matched nodes: ")
    print(nodes)


def read_batch(input_file):
    infer_list = np.loadtxt("./temp.txt", dtype=str, delimiter=",")
    return infer_list

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
    parser.add_argument('--rel', '-r',
                        metavar='rel',
                        default="",
                        type=str,
                        help='Relation, the original ID of a certain relation')
    parser.add_argument('--decoder', '-dc',
                        metavar='decoder',
                        default="DisMult",
                        choices=["DisMult", "TransE", "ComplEx"],
                        type=str,
                        help="Specifies the decoder used for training")
    parser.add_argument('--file_input', '-f',
                        metavar='file_input',
                        type=str,
                        help='File containing all required information for batch inference')

    return parser


def parse_infer_list(args):
    if args.file_input is None:
        if (args.dst is None) and (not args.src is None):
            infer_list = [[args.src, args.rel, ""]]
        elif (args.src is None) and (not args.dst is None):
            infer_list = [["", args.rel, args.dst]]
        else:
            raise RunTimeError("Incorrect source node or destination node.")
    else:
        infer_list = list(read_batch(args.file_input))
        if (args.dst is None) and (not args.src is None):
            infer_list.append([args.src, args.rel, ""])
        elif (args.src is None) and (not args.dst is None):
            infer_list.append(["", args.rel, args.dst])
    
    return infer_list


def main():
    parser = set_args()
    args = parser.parse_args()
    emb_dir = args.trained_embeddings_directory
    dataset_dir = args.dataset_directory
    infer_list = parse_infer_list(args)

    top_k_nodes = perform_link_prediction(emb_dir, dataset_dir, infer_list, int(args.k), args.decoder)
    print_nodes(top_k_nodes, args.k)

if __name__=='__main__':
    main()