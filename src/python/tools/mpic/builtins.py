from marius.tools.mpic.utils import Arg, Attrs, Callable, ClassType

IntType = ClassType("int")
StrType = ClassType("str")
VoidType = ClassType("None")

IntAttrs = Attrs({"_mpic_class": IntType})
StrAttrs = Attrs({"_mpic_class": StrType})
NoneAttrs = Attrs({"_mpic_class": VoidType})


def is_numeric(value_attrs: Attrs) -> bool:
    return value_attrs is IntAttrs


def is_consistent_with(src_attrs: Attrs, dst_attrs: Attrs) -> bool:
    """
    Return True iff objects of type src can be converted to objects of type dst
    """
    if src_attrs is NodeDataAttrs or src_attrs is EdgeDataAttrs:
        src_attrs = TensorAttrs
    return src_attrs is dst_attrs


# XXX: assumption: single graph in the entire program
node_data_prefix = "_mpic_ndata_"
edge_data_prefix = "_mpic_edata_"

Tensor = ClassType("_mpic_Tensor")  # torch Tensor
Linear = ClassType("_mpic_Linear")  # torch Linear
DENSEGraph = ClassType("_mpic_DENSEGraph")  # Marius DENSEGraph
Layer = ClassType("_mpic_Module")  # Marius base layer

MessageFuncAttrs = Attrs({"_mpic_class": "_mpic_MessageFunc"})
ReduceFuncAttrs = Attrs({"_mpic_class": "_mpic_ReduceFunc"})
NodeDataAttrs = Attrs({"_mpic_class": Tensor})
EdgeDataAttrs = Attrs({"_mpic_class": Tensor})
GraphLocalScopeAttrs = Attrs({"_mpic_class": "_mpic_GraphLocalScope"})

LayerAttrs = Attrs(
    {
        "__init__": Callable(
            [Arg("_mpic_input_dim", IntAttrs), Arg("_mpic_output_dim", IntAttrs)],
            NoneAttrs,
        ),
        "_mpic_class": Layer,
    }
)
TensorAttrs = Attrs({"_mpic_class": Tensor})
LinearAttrs = Attrs(
    {
        "__init__": Callable(
            [
                Arg("in_features", IntAttrs),
                Arg("out_features", IntAttrs),
            ],
            NoneAttrs,
        ),
        "reset_parameters": Callable([], NoneAttrs),
        "__call__": Callable([Arg("inputs", TensorAttrs)], TensorAttrs),
        "_mpic_class": Linear,
    }
)
DENSEGraphAttrs = Attrs(
    {
        "update_all": Callable(
            [
                Arg("message_func", MessageFuncAttrs),
                Arg("reduce_func", ReduceFuncAttrs),
            ],
            NoneAttrs,
        ),
        "local_scope": Callable([], GraphLocalScopeAttrs),
        "ndata": NodeDataAttrs,
        "edata": EdgeDataAttrs,
        "_mpic_class": DENSEGraph,
    }
)


def get_builtin_classes():
    # XXX: operator overriding is not supported!
    return {
        IntType: IntAttrs,
        StrType: StrAttrs,
        VoidType: NoneAttrs,
        Tensor: TensorAttrs,
        Linear: LinearAttrs,
        DENSEGraph: DENSEGraphAttrs,
        Layer: LayerAttrs,
    }


builtin_attrs = Attrs({"int": IntType, "str": StrType})
builtin_attrs["mpi"] = {
    # XXX: no inheritance supported => self is current type
    "Tensor": Tensor,
    "Linear": Linear,
    "DENSEGraph": DENSEGraph,
    "Module": Layer,
    "copy_u": Callable([Arg("u", StrAttrs), Arg("out", StrAttrs)], MessageFuncAttrs),
    "mean": Callable([Arg("msg", StrAttrs), Arg("out", StrAttrs)], ReduceFuncAttrs),
    "cat": Callable(
        [
            Arg("tensor0", TensorAttrs),
            Arg("tensor1", TensorAttrs),
            Arg("dim", IntAttrs),
        ],
        TensorAttrs,
    ),
}


def get_builtin_typemap() -> dict[Attrs, str]:
    return {
        IntType: "int",
        StrType: "string",
        Tensor: "torch::Tensor",
        Linear: "torch::Tensor",
        DENSEGraph: "DENSEGraph",
    }
