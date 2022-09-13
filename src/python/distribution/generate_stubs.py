import os

from pybind11_stubgen import ModuleStubsGenerator


def generate_stubs(output_dir, module_name):
    module = ModuleStubsGenerator(module_name)

    module.parse()

    module.write_setup_py = False

    module_name = module_name.split(".")[-1]

    os.makedirs(output_dir, exist_ok=True)

    with open("{}/{}.pyi".format(output_dir, module_name), "w") as fp:
        fp.write("#\n# AUTOMATICALLY GENERATED FILE\n#\n\n")
        fp.write("import torch\n")
        fp.write("\n".join(module.to_lines()))


def gen_all_stubs(output_dir):
    generate_stubs(output_dir, "marius.config")

    generate_stubs(output_dir, "marius._data.samplers")
    generate_stubs(output_dir, "marius.data")

    generate_stubs(output_dir, "marius.manager")

    generate_stubs(output_dir, "marius._nn.decoders.edge")
    generate_stubs(output_dir, "marius._nn.decoders.node")
    generate_stubs(output_dir, "marius._nn.decoders")
    generate_stubs(output_dir, "marius._nn.encoders")
    generate_stubs(output_dir, "marius._nn.layers")
    generate_stubs(output_dir, "marius.nn")

    generate_stubs(output_dir, "marius.pipeline")
    generate_stubs(output_dir, "marius.report")
    generate_stubs(output_dir, "marius.storage")

    generate_stubs(output_dir, "marius")


if __name__ == "__main__":
    gen_all_stubs("tmp")
# generate_stubs("tmp", "marius")
# generate_stubs("tmp", "marius.config")
# generate_stubs("tmp", "marius.data")
# generate_stubs("tmp", "marius._data.samplers")
# generate_stubs("tmp", "marius.manager")
# generate_stubs("tmp", "marius.nn")
# generate_stubs("tmp", "marius._nn.decoders")
# generate_stubs("tmp", "marius._nn.encoders")
# generate_stubs("tmp", "marius._nn.layers")
# generate_stubs("tmp", "marius.pipeline")
# generate_stubs("tmp", "marius.report")
# generate_stubs("tmp", "marius.storage")

# generate_stubs("tmp", "marius")
# generate_stubs("tmp", "marius.config")
# generate_stubs("tmp", "marius.data")
# generate_stubs("tmp", "marius.data.samplers")
# generate_stubs("tmp", "marius.manager")
# generate_stubs("tmp", "marius.nn")
# generate_stubs("tmp", "marius.nn.decoders")
# generate_stubs("tmp", "marius.nn.encoders")
# generate_stubs("tmp", "marius.nn.layers")
# generate_stubs("tmp", "marius.pipeline")
# generate_stubs("tmp", "marius.report")
# generate_stubs("tmp", "marius.storage")
