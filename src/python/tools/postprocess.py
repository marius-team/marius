from pathlib import Path
import numpy as np
import torch
import argparse


def get_embs(mapping_file, embs_file):
    try:
        mapping = np.loadtxt(mapping_file, dtype=str, delimiter='\t')
        mapping_dict = dict(mapping)

        num = len(mapping_dict)
        embs = np.fromfile(embs_file, np.float32).reshape((num, -1))
    except FileNotFoundError:
        raise FileNotFoundError("Incorrect file passed in.")

    return embs


def output_embeddings(data_dir, dataset_dir, output_dir, fmt):
    node_mapping_file = Path(dataset_dir) / Path("node_mapping.txt")
    rel_mapping_file = Path(dataset_dir) / Path("rel_mapping.txt")
    node_embs_file = Path(data_dir) / Path("embeddings/embeddings.bin")
    lhs_embs_file = Path(data_dir) / Path("relations/src_relations.bin")
    rhs_embs_file = Path(data_dir) / Path("relations/dst_relations.bin")

    node_embs = get_embs(node_mapping_file, node_embs_file)
    lhs_embs = get_embs(rel_mapping_file, lhs_embs_file)
    rhs_embs = get_embs(rel_mapping_file, rhs_embs_file)

    if fmt == "CSV":
        np.savetxt(Path(output_dir) / Path("node_embeddings.csv"), node_embs, delimiter=',')
        np.savetxt(Path(output_dir) / Path("src_relations_embeddings.csv"), lhs_embs, delimiter=',')
        np.savetxt(Path(output_dir) / Path("dst_relations_embeddings.csv"), rhs_embs, delimiter=',')
    elif fmt == "TSV":
        np.savetxt(Path(output_dir) / Path("node_embeddings.tsv"), node_embs, delimiter='\t')
        np.savetxt(Path(output_dir) / Path("src_relations_embeddings.tsv"), lhs_embs, delimiter='\t')
        np.savetxt(Path(output_dir) / Path("dst_relations_embeddings.tsv"), rhs_embs, delimiter='\t')
    elif fmt == "TXT":
        np.savetxt(Path(output_dir) / Path("node_embeddings.txt"), node_embs, delimiter='\t')
        np.savetxt(Path(output_dir) / Path("src_relations_embeddings.txt"), lhs_embs, delimiter='\t')
        np.savetxt(Path(output_dir) / Path("dst_relations_embeddings.txt"), rhs_embs, delimiter='\t')
    else:
        torch.save(torch.tensor(node_embs), Path(output_dir) / Path('node_embeddings.pt'))
        torch.save(torch.tensor(lhs_embs), Path(output_dir) / Path('src_relations_embeddings.pt'))
        torch.save(torch.tensor(rhs_embs), Path(output_dir) / Path('dst_relations_embeddings.pt'))

    return node_embs, lhs_embs, rhs_embs


def set_args():
    parser = argparse.ArgumentParser(
        description='Retrieve trained embeddings',
        prog='postprocess'
    )
    parser.add_argument('trained_embeddings_directory',
                        metavar='trained_embeddings_directory',
                        type=str,
                        help='Directory containing trained embeddings')
    parser.add_argument('dataset_directory',
                        metavar='dataset_directory',
                        type=str,
                        help='Directory containing the dataset for training')
    parser.add_argument('--output_directory', '-o',
                        metavar='output_directory',
                        type=str,
                        help='Directory to put retrieved embeddings. ' + 
                             'If is not set, will output retrieved embeddings' +
                             ' to dataset directory.')
    parser.add_argument('--format', '-f',
                        metavar='format',
                        choices=["CSV", "TSV", "TXT", "Tensor"],
                        default="CSV",
                        help="Data format to store retrieved embeddings")

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()
    data_dir = args.trained_embeddings_directory
    dataset_dir = args.dataset_directory
    output_dir = args.output_directory
    fmt = args.format

    if output_dir is None:
        output_dir = dataset_dir
    else:
        if not Path(output_dir).exists():
            Path(output_dir).mkdir()

    output_embeddings(data_dir, dataset_dir, output_dir, fmt)


if __name__ == '__main__':
    main()
