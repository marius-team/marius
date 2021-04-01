import argparse
import os
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))) + "/tools")
import tools.preprocess as preprocess
import execute as e
import json


def run_fb15k():
    exp_dir = "osdi2021/system_comparisons/fb15k/marius/"

    distmult_config = exp_dir + "distmult.ini"
    complex_config = exp_dir + "complex.ini"

    if not os.path.exists("fb15k/"):
        print("==== Preprocessing FB15K =====")
        preprocess.fb15k("fb15k/")

    num_runs = 5

    distmult_info_logs = []
    complex_info_logs = []

    print("==== Running Marius: DistMult FB15K =====")
    for i in range(num_runs):
        args = "--general.random_seed=%i" % i
        e.run_marius(distmult_config, args)
        info_log, _, _ = e.collect_metrics(info_log_only=True)
        info_log["Epoch Progress"] = []
        distmult_info_logs.append(info_log)
        e.cleanup_experiments(info_log_only=True)

    print("==== Running Marius: ComplEx FB15K =====")
    for i in range(num_runs):
        args = "--general.random_seed=%i" % i
        e.run_marius(complex_config, args)
        info_log, _, _ = e.collect_metrics(info_log_only=True)
        info_log["Epoch Progress"] = []
        complex_info_logs.append(info_log)
        e.cleanup_experiments(info_log_only=True)

    with open(exp_dir + "distmult_result.json", 'w') as dm_out:
        json.dump(distmult_info_logs, dm_out)

    with open(exp_dir + "complex_result.json", 'w') as cx_out:
        json.dump(distmult_info_logs, cx_out)


def run_livejournal():
    exp_dir = "osdi2021/system_comparisons/livejournal/marius"

    dot_config = exp_dir + "dot.ini"

    if not os.path.exists("livejournal/"):
        print("==== Preprocessing Livejournal =====")
        preprocess.twitter("livejournal/")

    dot_info_logs = []

    print("==== Running Marius: Dot Livejournal =====")
    args = "--general.random_seed=%i" % 0
    e.run_marius(dot_config, args)
    info_log, _, _ = e.collect_metrics(info_log_only=True)
    dot_info_logs.append(info_log)
    e.cleanup_experiments(info_log_only=True)

    with open(exp_dir + "dot_result.json", 'w') as cx_out:
        json.dump(dot_info_logs, cx_out)


def run_twitter():
    exp_dir = "osdi2021/system_comparisons/twitter/marius"

    dot_config = exp_dir + "dot.ini"

    if not os.path.exists("twitter/"):
        print("==== Preprocessing Twitter =====")
        preprocess.twitter("twitter/")

    dot_info_logs = []

    print("==== Running Marius: Dot Twitter =====")
    args = "--general.random_seed=%i" % 0
    e.run_marius(dot_config, args)
    info_log, _, _ = e.collect_metrics(info_log_only=True)
    dot_info_logs.append(info_log)
    e.cleanup_experiments(info_log_only=True)

    with open(exp_dir + "dot_result.json", 'w') as cx_out:
        json.dump(dot_info_logs, cx_out)


def run_freebase86m():
    exp_dir = "osdi2021/system_comparisons/freebase86m/marius/"
    complex_config = exp_dir + "d100.ini"

    if not os.path.exists("freebase86m_p16/"):
        print("==== Preprocessing Freebase86m P=16 D=100 =====")
        preprocess.freebase86m("freebase86m_p16/", num_partitions=16)

    complex_info_logs = []

    print("==== Running Marius: ComplEx Freebase86m D=100 =====")
    args = "--general.random_seed=%i" % 0
    e.run_marius(complex_config, args)
    info_log, _, _ = e.collect_metrics(info_log_only=True)
    complex_info_logs.append(info_log)
    e.cleanup_experiments(info_log_only=True)

    with open(exp_dir + "complex_100.json", 'w') as cx_out:
        json.dump(complex_info_logs, cx_out)


def run_utilization():

    exp_dir = "osdi2021/system_comparisons/freebase86m/marius/"

    complex_50_config = exp_dir + "d50.ini"
    complex_100_config = exp_dir + "d100.ini"

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    complex_50_info_logs = []
    print("==== Running Marius: ComplEx Freebase86m D=50 =====")
    args = "--general.random_seed=%i" % 0
    dstat_pid, nvidia_smi_pid = e.start_tracing()
    e.run_marius(complex_50_config, args)
    e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
    info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
    complex_50_info_logs.append(info_log)
    e.cleanup_experiments()

    with open(exp_dir + "complex_50_result.json", 'w') as cx_out:
        json.dump(complex_50_info_logs, cx_out)
    with open(exp_dir + "complex_50_dstat.csv", 'w') as cx_out:
        dstat_df.to_csv(cx_out)
    with open(exp_dir + "complex_50_nvidia_smi.csv", 'w') as cx_out:
        nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("freebase86m_p16/"):
        print("==== Preprocessing Freebase86m P=16 D=100 =====")
        preprocess.freebase86m("freebase86m_p16/", num_partitions=16)

    complex_100_info_logs = []
    print("==== Running Marius: ComplEx Freebase86m D=100 =====")
    args = "--general.random_seed=%i" % 0
    dstat_pid, nvidia_smi_pid = e.start_tracing()
    e.run_marius(complex_100_config, args)
    e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
    info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
    complex_100_info_logs.append(info_log)
    e.cleanup_experiments()

    with open(exp_dir + "complex_100_result.json", 'w') as cx_out:
        json.dump(complex_100_info_logs, cx_out)
    with open(exp_dir + "complex_100_dstat.csv", 'w') as cx_out:
        dstat_df.to_csv(cx_out)
    with open(exp_dir + "complex_100_nvidia_smi.csv", 'w') as cx_out:
        nvidia_smi_df.to_csv(cx_out)


def run_buffer_simulator():
    pass


def run_orderings_total_io():

    exp_dir = "osdi2021/system_comparisons/partition_orderings/freebase86m/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"


    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)


    if not os.path.exists("elimination100_result.json"):
        print("==== Running Marius: Elimination Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(elimination_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "elimination100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "elimination100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "elimination100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbert100_result.json"):
        print("==== Running Marius: Hilbert Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbert100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbert100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbert100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbertsymmetric100_result.json"):
        print("==== Running Marius: HilbertSymmetric Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_symmetric_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbertsymmetric100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbertsymmetric100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbertsymmetric100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)


def run_orderings_freebase86m():
    exp_dir = "osdi2021/system_comparisons/partition_orderings/freebase86m/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"
    memory_config = exp_dir + "memory.ini"

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    if not os.path.exists("elimination100_result.json"):
        print("==== Running Marius: Elimination Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(elimination_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "elimination100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "elimination100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "elimination100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbert100_result.json"):
        print("==== Running Marius: Hilbert Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbert100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbert100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbert100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbertsymmetric100_result.json"):
        print("==== Running Marius: HilbertSymmetric Freebase86m D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_symmetric_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbertsymmetric100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbertsymmetric100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbertsymmetric100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("elimination50_result.json"):
        print("==== Running Marius: Elimination Freebase86m D=50 =====")
        args = "--general.random_seed=%i --model.embedding_size=50" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(elimination_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "elimination50_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "elimination50_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "elimination50_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbert50_result.json"):
        print("==== Running Marius: Hilbert Freebase86m D=50 =====")
        args = "--general.random_seed=%i --model.embedding_size=50" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbert50_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbert50_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbert50_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbertsymmetric50_result.json"):
        print("==== Running Marius: HilbertSymmetric Freebase86m D=50 =====")
        args = "--general.random_seed=%i --model.embedding_size=50" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_symmetric_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbertsymmetric50_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbertsymmetric50_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbertsymmetric50_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("memory50_result.json"):
        print("==== Running Marius: Memory Freebase86m D=50 =====")
        args = "--general.random_seed=%i --model.embedding_size=50" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(memory_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "memory50_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "memory50_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "memory50_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)


def run_orderings_twitter():
    exp_dir = "osdi2021/system_comparisons/partition_orderings/twitter/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"
    memory_config = exp_dir + "memory.ini"

    if not os.path.exists("twitter_32/"):
        print("==== Preprocessing Twitter P=32 D=200 =====")
        preprocess.freebase86m("twitter_32/", num_partitions=32)

    if not os.path.exists("twitter/"):
        print("==== Preprocessing Twitter P=1 D=100 =====")
        preprocess.freebase86m("twitter/")

    if not os.path.exists("elimination100_result.json"):
        print("==== Running Marius: Elimination Twitter D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(elimination_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "elimination100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "elimination100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "elimination100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbert100_result.json"):
        print("==== Running Marius: Hilbert Twitter D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbert100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbert100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbert100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbertsymmetric100_result.json"):
        print("==== Running Marius: HilbertSymmetric Twitter D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_symmetric_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbertsymmetric100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbertsymmetric100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbertsymmetric100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("memory50_result.json"):
        print("==== Running Marius: Memory Twitter D=100 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(memory_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "memory100_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "memory100_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "memory100_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("elimination200_result.json"):
        print("==== Running Marius: Elimination Twitter D=200 =====")
        args = "--general.random_seed=%i --model.embedding_size=200" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(elimination_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "elimination200_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "elimination200_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "elimination200_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbert200_result.json"):
        print("==== Running Marius: Hilbert Twitter D=200 =====")
        args = "--general.random_seed=%i --model.embedding_size=200" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbert200_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbert200_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbert200_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("hilbertsymmetric200_result.json"):
        print("==== Running Marius: HilbertSymmetric Twitter D=200 =====")
        args = "--general.random_seed=%i --model.embedding_size=200" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(hilbert_symmetric_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "hilbertsymmetric200_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "hilbertsymmetric200_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "hilbertsymmetric200_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)


def run_staleness_bound():
    exp_dir = "osdi2021/system_comparisons/microbenchmarks/bounded_staleness"

    all_async_config = exp_dir + "all_async.ini"
    all_sync = exp_dir + "all_sync.ini"
    sync_relations_async_nodes = exp_dir + "sync_relations_async_nodes.ini"

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    if not os.path.exists("sync_result.json"):
        print("==== Running Marius: Sync Freebase86m D=50 =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(all_sync, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "sync_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "sync_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "sync_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("sync_relations_bound_2_result.json"):
        for bound in [2, 4, 8, 16, 32, 64]:
            print("==== Running Marius: Sync Relations, Bound=%i Freebase86m D=50 =====" % bound)
            args = "--general.random_seed=%i --training_pipeline.max_batches_in_flight=%i" % (0, bound)
            dstat_pid, nvidia_smi_pid = e.start_tracing()
            e.run_marius(sync_relations_async_nodes, args)
            e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
            info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
            e.cleanup_experiments()

            with open(exp_dir + "sync_relations_bound_2_result.json", 'w') as cx_out:
                json.dump(info_log, cx_out)
            with open(exp_dir + "sync_relations_bound_2_dstat.csv", 'w') as cx_out:
                dstat_df.to_csv(cx_out)
            with open(exp_dir + "sync_relations_bound_2_nvidia_smi.csv", 'w') as cx_out:
                nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("async_relations_bound_2_result.json"):
        for bound in [2, 4, 8, 16, 32, 64]:
            print("==== Running Marius: Async Relations, Bound=%i Freebase86m D=50 =====" % bound)
            args = "--general.random_seed=%i --training_pipeline.max_batches_in_flight=%i" % (0, bound)
            dstat_pid, nvidia_smi_pid = e.start_tracing()
            e.run_marius(all_async_config, args)
            e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
            info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
            e.cleanup_experiments()

            with open(exp_dir + "async_relations_bound_2_result.json", 'w') as cx_out:
                json.dump(info_log, cx_out)
            with open(exp_dir + "async_relations_bound_2_dstat.csv", 'w') as cx_out:
                dstat_df.to_csv(cx_out)
            with open(exp_dir + "async_relations_bound_2_nvidia_smi.csv", 'w') as cx_out:
                nvidia_smi_df.to_csv(cx_out)


def run_prefetching():
    exp_dir = "osdi2021/system_comparisons/microbenchmarks/prefetching"

    no_prefetching_config = exp_dir + "no_prefetching.ini"
    prefetching_config = exp_dir + "prefetching.ini"

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/")

    if not os.path.exists("no_prefetching_result.json"):
        print("==== Running Marius: No Prefetching Freebase86m =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(no_prefetching_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "no_prefetching_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "no_prefetching_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "no_prefetching_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)

    if not os.path.exists("prefetching_result.json"):
        print("==== Running Marius: Prefetching Freebase86m =====")
        args = "--general.random_seed=%i" % 0
        dstat_pid, nvidia_smi_pid = e.start_tracing()
        e.run_marius(prefetching_config, args)
        e.stop_metric_collection(dstat_pid, nvidia_smi_pid)
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics()
        e.cleanup_experiments()

        with open(exp_dir + "prefetching_result.json", 'w') as cx_out:
            json.dump(info_log, cx_out)
        with open(exp_dir + "prefetching_dstat.csv", 'w') as cx_out:
            dstat_df.to_csv(cx_out)
        with open(exp_dir + "prefetching_nvidia_smi.csv", 'w') as cx_out:
            nvidia_smi_df.to_csv(cx_out)


def run_big_embeddings():
    pass


def run_all():
    pass


if __name__ == "__main__":
    experiment_dict = {
        "fb15k": run_fb15k,
        "livejournal": run_livejournal,
        "twitter": run_twitter,
        "freebase86m": run_freebase86m,
        "utilization": run_utilization,
        "buffer_simulator": run_buffer_simulator,
        "orderings_total_io": run_orderings_total_io,
        "orderings_freebase86m": run_orderings_freebase86m,
        "orderings_twitter": run_orderings_twitter,
        "prefetching": run_prefetching,
        "big_embeddings": run_big_embeddings,
        "all": run_all
    }
    parser = argparse.ArgumentParser(description='Reproduce experiments ')
    parser.add_argument('--experiment', metavar='experiment', type=str, choices=experiment_dict.keys(),
                        help='Experiment choices: %(choices)s')

    args = parser.parse_args()
    experiment_dict.get(args.experiment)()
