import argparse
import os
import sys
from os import path
import execute as e
import json
from buffer_simulator import plotting as plot_buff
import plotting as osdi_plot
import shutil
import marius.tools.preprocess as preprocess
from dglke_preprocessing import preprocess_live_journal
import itertools

def run_marius(config, exp_dir, name, config_args="", overwrite=False, collect_tracing_metrics=False, show_output=False):
    e.cleanup_experiments()
    if not os.path.exists(exp_dir + name + "_result.json") or overwrite:
        print("==== Running Marius: %s =====" % name)

        dstat_pid = None
        nvidia_smi_pid = None
        if collect_tracing_metrics:
            dstat_pid, nvidia_smi_pid = e.start_tracing()

        try:
            e.run_marius(config, config_args, show_output=show_output)
        except Exception as ex:
            print("Run Failed: %s" % ex)

            if collect_tracing_metrics:
                e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

            e.cleanup_experiments()
            return

        if collect_tracing_metrics:
            e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

        info_log_only = not collect_tracing_metrics
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics(info_log_only=info_log_only)

        with open(exp_dir + name + "_result.json", 'w') as out_file:
            json.dump(info_log, out_file)
            print("Marius output results written to: %s" % exp_dir + name + "_result.json")

        if collect_tracing_metrics:
            with open(exp_dir + name + "_dstat.csv", 'w') as out_file:
                dstat_df.to_csv(out_file)
                print("Dstat tracing results written to: %s" % exp_dir + name + "_dstat.csv")
            with open(exp_dir + name + "_nvidia_smi.csv", 'w') as out_file:
                nvidia_smi_df.to_csv(out_file)
                print("Nvidia-smi tracing results written to: %s" % exp_dir + name + "_nvidia_smi.csv")

        shutil.move("tmp.txt", exp_dir + name + "_output.txt")
        print("Program output written to: %s" % exp_dir + name + "_output.txt")

        e.cleanup_experiments()
    else:
        print("Marius: %s already run" % name)


def run_pbg(runner_file, config, exp_dir, name, config_args="", overwrite=False, collect_tracing_metrics=False, show_output=False, eval_in_marius=False):
    e.cleanup_experiments()
    if not os.path.exists(exp_dir + name + "_result.json") or overwrite:
        print("==== Running PBG: %s =====" % name)

        dstat_pid = None
        nvidia_smi_pid = None
        if collect_tracing_metrics:
            dstat_pid, nvidia_smi_pid = e.start_tracing()
        try:
            e.run_pbg(runner_file, config, config_args, show_output=show_output)
        except Exception as ex:
            print("Run Failed: %s" % ex)

            if collect_tracing_metrics:
                e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

            e.cleanup_experiments()
            return

        if collect_tracing_metrics:
            e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

        info_log_only = not collect_tracing_metrics
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics(info_log_only=info_log_only, pbg=True, eval_in_marius=eval_in_marius, experiment_name=name)

        with open(exp_dir + name + "_result.json", 'w') as out_file:
            json.dump(info_log, out_file)
            print("PBG output results written to: %s" % exp_dir + name + "_result.json")
        if collect_tracing_metrics:
            with open(exp_dir + name + "_dstat.csv", 'w') as out_file:
                dstat_df.to_csv(out_file)
                print("Dstat tracing results written to: %s" % exp_dir + name + "_dstat.csv")
            with open(exp_dir + name + "_nvidia_smi.csv", 'w') as out_file:
                nvidia_smi_df.to_csv(out_file)
                print("Nvidia-smi tracing results written to: %s" % exp_dir + name + "_nvidia_smi.csv")

        shutil.move("tmp.txt", exp_dir + name + "_output.txt")
        print("Program output written to: %s" % exp_dir + name + "_output.txt")

        e.cleanup_experiments()
    else:
        print("PBG: %s already run" % name)


def run_dglke(cmd, exp_dir, name, overwrite=False, collect_tracing_metrics=False, show_output=False):
    e.cleanup_experiments()
    if not os.path.exists(exp_dir + name + "_result.json") or overwrite:
        print("==== Running DGL-KE: %s =====" % name)

        dstat_pid = None
        nvidia_smi_pid = None
        if collect_tracing_metrics:
            dstat_pid, nvidia_smi_pid = e.start_tracing()

        try:
            e.run_dglke(cmd, show_output=show_output)
        except Exception as ex:
            print("Run Failed: %s" % ex)

            if collect_tracing_metrics:
                e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

            e.cleanup_experiments()
            return

        if collect_tracing_metrics:
            e.stop_metric_collection(dstat_pid, nvidia_smi_pid)

        info_log_only = not collect_tracing_metrics
        info_log, dstat_df, nvidia_smi_df = e.collect_metrics(info_log_only=info_log_only, dglke=True)

        with open(exp_dir + name + "_result.json", 'w') as out_file:
            json.dump(info_log, out_file)
            print("DGL-KE output results written to: %s" % exp_dir + name + "_result.json")

        if collect_tracing_metrics:
            with open(exp_dir + name + "_dstat.csv", 'w') as out_file:
                dstat_df.to_csv(out_file)
                print("Dstat tracing results written to: %s" % exp_dir + name + "_dstat.csv")
            with open(exp_dir + name + "_nvidia_smi.csv", 'w') as out_file:
                nvidia_smi_df.to_csv(out_file)
                print("Nvidia-smi tracing results written to: %s" % exp_dir + name + "_nvidia_smi.csv")

        shutil.move("tmp.txt", exp_dir + name + "_output.txt")
        print("Program output written to: %s" % exp_dir + name + "_output.txt")

        e.cleanup_experiments()
    else:
        print("DGL-KE: %s already run" % name)


def run_fb15k(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/system_comparisons/fb15k/marius/"

    distmult_config = exp_dir + "distmult.ini"
    complex_config = exp_dir + "complex.ini"

    if not os.path.exists("fb15k/"):
        print("==== Preprocessing FB15K =====")
        preprocess.fb15k("fb15k/")

    run_marius(distmult_config, exp_dir, "distmult_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(complex_config, exp_dir, "complex_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/fb15k/dgl-ke/"
    with open(exp_dir + "complex.txt", "r") as f:
        dglke_complex_cmd = f.readlines()[0]
    with open(exp_dir + "distmult.txt", "r") as f:
        dglke_distmult_cmd = f.readlines()[0]

    run_dglke(dglke_complex_cmd, exp_dir, "complex_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_dglke(dglke_distmult_cmd, exp_dir, "distmult_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    # PBG may throw errors
    exp_dir = "osdi2021/system_comparisons/fb15k/pbg/"

    runner_file = exp_dir + "run_fb15k.py"
    complex_config = exp_dir + "fb15k_complex_config.py"
    distmult_config = exp_dir + "fb15k_distmult_config.py"
    run_pbg(runner_file, complex_config, exp_dir, "complex_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_pbg(runner_file, distmult_config, exp_dir, "distmult_fb15k", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.print_table_2()


def run_livejournal(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/system_comparisons/livejournal/marius/"
    dot_config = exp_dir + "dot.ini"

    if not os.path.exists("livejournal/"):
        print("==== Preprocessing Livejournal =====")
        preprocess.live_journal("livejournal/")

    if not os.path.exists("live_journal_dglke/"):
        print("==== Preprocessing Livejournal DGL-KE =====")
        preprocess_live_journal.preproccess_live_journal()

    run_marius(dot_config, exp_dir, "dot_live_journal", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/livejournal/pbg/"

    runner_file = exp_dir + "run_dot.py"
    dot_config = exp_dir + "dot.py"
    run_pbg(runner_file, dot_config, exp_dir, "dot_live_journal", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output, eval_in_marius=True)

    exp_dir = "osdi2021/system_comparisons/livejournal/dgl-ke/"

    with open(exp_dir + "dot.txt", "r") as f:
        dglke_dot_cmd = f.readlines()[0]

    run_dglke(dglke_dot_cmd, exp_dir, "dot_live_journal", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.print_table_3()


def run_twitter(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/system_comparisons/twitter/marius/"

    dot_config = exp_dir + "dot.ini"

    if not os.path.exists("twitter_8/"):
        print("==== Preprocessing Twitter =====")
        preprocess.twitter("twitter_8/", num_partitions=8)

    run_marius(dot_config, exp_dir, "dot_twitter", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/twitter/pbg/"

    runner_file = exp_dir + "run_twitter.py"
    dot_config = exp_dir + "twitter_16.py"
    run_pbg(runner_file, dot_config, exp_dir, "dot_twitter", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output, eval_in_marius=True)

    osdi_plot.print_table_4()


def run_freebase86m(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/system_comparisons/freebase86m/marius/"
    complex_config = exp_dir + "d100.ini"

    if not os.path.exists("freebase86m_p16/"):
        print("==== Preprocessing Freebase86m P=16 D=100 =====")
        preprocess.freebase86m("freebase86m_p16/", num_partitions=16)

    # if not os.path.exists("freebase86m_p8/"):
    #     print("==== Preprocessing Freebase86m P=8 D=100 =====")
    #     preprocess.freebase86m("freebase86m_p8/", num_partitions=8)

    run_marius(complex_config, exp_dir, "freebase86m_16", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/freebase86m/pbg/"

    runner_file = exp_dir + "run_complex.py"
    pbg_complex_config = exp_dir + "complex_p16.py"
    run_pbg(runner_file, pbg_complex_config, exp_dir, "freebase86m_16", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output, eval_in_marius=True)

    osdi_plot.print_table_5()


def run_utilization(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/system_comparisons/freebase86m/marius/"

    complex_50_config = exp_dir + "d50.ini"
    complex_50_8_config = exp_dir + "d50_8.ini"

    # this experiment requires collecting metrics
    collect_tracing_metrics = True

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    config_args = "--training.num_epochs=1 --evaluation.epochs_per_eval=2"
    run_marius(complex_50_config, exp_dir, "complex_50_util", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    if not os.path.exists("freebase86m_p8/"):
        print("==== Preprocessing Freebase86m P=8 D=50 =====")
        preprocess.freebase86m("freebase86m_p8/", num_partitions=8)

    run_marius(complex_50_8_config, exp_dir, "complex_50_8_util", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    # exp_dir = "osdi2021/system_comparisons/freebase86m/dgl-ke/"
    # with open(exp_dir + "complex.txt", "r") as f:
    #     dglke_complex_cmd = f.readlines()[0]
    #
    # run_dglke(dglke_complex_cmd, exp_dir, "complex_50_util", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/freebase86m/pbg/"

    runner_file = exp_dir + "run_complex.py"
    dot_config = exp_dir + "complex_p8.py"
    run_pbg(runner_file, dot_config, exp_dir, "complex_50_8_util", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.plot_figure_8()


def run_buffer_simulator(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/buffer_simulator/"

    n_start = 8
    c_start = 2
    num = 5
    total_size = 86E6 * 4 * 2 * 100 # total embedding size for freebase86m d=100 + optimizer state
    plot_buff.plot_varying_num_partitions_io(n_start, c_start, num, total_size, exp_dir + "figure7.png")
    print("Figure written to %s" % exp_dir + "figure7.png")


def run_orderings_total_io(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/partition_orderings/freebase86m/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"

    # this experiment requires collecting metrics
    collect_tracing_metrics = True

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)

    config_args = "--training.num_epochs=1 --evaluation.epochs_per_eval=2"
    run_marius(elimination_config, exp_dir, "elimination100_util", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_config, exp_dir, "hilbert100_util", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_symmetric_config, exp_dir, "hilbertsymmetric100_util", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.plot_figure_9()


def run_orderings_freebase86m(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/partition_orderings/freebase86m/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"
    memory_config = exp_dir + "memory.ini"

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)

    config_args = ""
    if short:
        config_args += " --training.num_epochs=1"

    run_marius(elimination_config, exp_dir, "elimination100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_config, exp_dir, "hilbert100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_symmetric_config, exp_dir, "hilbertsymmetric100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    config_args = "--model.embedding_size=50"

    if short:
        config_args += " --training.num_epochs=1"

    run_marius(elimination_config, exp_dir, "elimination50", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_config, exp_dir, "hilbert50", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_symmetric_config, exp_dir, "hilbertsymmetric50", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(memory_config, exp_dir, "memory50", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.plot_figure_10()


def run_orderings_twitter(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/partition_orderings/twitter/"

    elimination_config = exp_dir + "elimination.ini"
    hilbert_config = exp_dir + "hilbert.ini"
    hilbert_symmetric_config = exp_dir + "hilbert_symmetric.ini"

    if not os.path.exists("twitter_32/"):
        print("==== Preprocessing Twitter P=32 =====")
        preprocess.twitter("twitter_32/", num_partitions=32)

    config_args = ""
    if short:
        config_args += " --training.num_epochs=1"

    run_marius(hilbert_symmetric_config, exp_dir, "hilbertsymmetric100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(elimination_config, exp_dir, "elimination100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_config, exp_dir, "hilbert100", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    config_args = "--model.embedding_size=200"
    if short:
        config_args += " --training.num_epochs=1"

    run_marius(elimination_config, exp_dir, "elimination200", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_config, exp_dir, "hilbert200", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(hilbert_symmetric_config, exp_dir, "hilbertsymmetric200", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.plot_figure_11()


def run_staleness_bound(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/microbenchmarks/bounded_staleness/"

    all_async_config = exp_dir + "all_async.ini"
    all_sync = exp_dir + "all_sync.ini"
    sync_relations_async_nodes = exp_dir + "sync_relations_async_nodes.ini"

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 D=50 =====")
        preprocess.freebase86m("freebase86m/")

    config_args = ""
    if short:
        config_args += " --training.num_epochs=3"

    run_marius(all_sync, exp_dir, "all_sync", config_args=config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    bounds = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    if short:
        bounds = [4, 16, 64, 256]

    for bound in bounds:
        config_args = "--training_pipeline.max_batches_in_flight=%i" % bound

        if short:
            config_args += " --training.num_epochs=3"

        run_marius(all_async_config, exp_dir, "all_async_%i" % bound, config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
        run_marius(sync_relations_async_nodes, exp_dir, "sync_rel_%i" % bound, config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.plot_figure_12(bounds)


def run_prefetching(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/microbenchmarks/prefetching/"

    no_prefetching_config = exp_dir + "no_prefetching.ini"
    prefetching_config = exp_dir + "prefetching.ini"

    # this experiment requires collecting metrics
    collect_tracing_metrics = True

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 D=100 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)

    run_marius(no_prefetching_config, exp_dir, "no_prefetching", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(prefetching_config, exp_dir, "prefetching", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.print_figure_13()


def run_big_embeddings(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/large_embeddings/"
    cpu_memory = exp_dir + "cpu_memory.ini"
    gpu_memory = exp_dir + "gpu_memory.ini"
    disk = exp_dir + "disk.ini"
    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m P=1 =====")
        preprocess.freebase86m("freebase86m/")

    if not os.path.exists("freebase86m_32/"):
        print("==== Preprocessing Freebase86m P=32 =====")
        preprocess.freebase86m("freebase86m_32/", num_partitions=32)

    run_marius(gpu_memory, exp_dir, "d20", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)
    run_marius(cpu_memory, exp_dir, "d50", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    config_args = "--storage.buffer_capacity=16"
    run_marius(disk, exp_dir, "d100", config_args, overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    osdi_plot.print_table_6()

    # config_args = "--storage.buffer_capacity=4 --model.embedding_size=400"
    # run_marius(disk, exp_dir, "d400", config_args)
    #
    # if not os.path.exists("freebase86m_64/"):
    #     print("==== Preprocessing Freebase86m P=64 =====")
    #     preprocess.freebase86m("freebase86m_64/", num_partitions=64)

    # config_args = "--storage.buffer_capacity=4 --model.embedding_size=800"
    # run_marius(disk, exp_dir, "d800", config_args)


def run_all(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    experiments = {
        "fb15k": run_fb15k,
        "livejournal": run_livejournal,
        "twitter": run_twitter,
        "freebase86m": run_freebase86m,
        "utilization": run_utilization,
        "buffer_simulator": run_buffer_simulator,
        "orderings_total_io": run_orderings_total_io,
        "orderings_freebase86m": run_orderings_freebase86m,
        "orderings_twitter": run_orderings_twitter,
        "staleness_bound": run_staleness_bound,
        "prefetching": run_prefetching,
        "big_embeddings": run_big_embeddings
    }

    print("#### Running all experiments ####\n")
    for k, v in experiments.items():
        print("#### Running %s ####\n" %k)
        try:
            v(overwrite, collect_tracing_metrics, show_output, short)
            print("#### %s Complete ####\n" %k)
        except Exception as e:
            print(e)
            print("#### %s Failed ####\n" %k)


def run_multi_gpu(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):

    exp_dir = "osdi2021/system_comparisons/freebase86m/marius/"
    complex_config = exp_dir + "d50_multi_gpu.ini"

    # if not os.path.exists("freebase86m_p16/"):
    #     print("==== Preprocessing Freebase86m P=16 D=100 =====")
    #     preprocess.freebase86m("freebase86m_p16/", num_partitions=16)

    if not os.path.exists("freebase86m/"):
        print("==== Preprocessing Freebase86m =====")
        preprocess.freebase86m("freebase86m/")

    run_marius(complex_config, exp_dir, "complex_8gpu", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

    exp_dir = "osdi2021/system_comparisons/freebase86m/dgl-ke/"

    # with open(exp_dir + "complex_multi_gpu.txt", "r") as f:
    #     dglke_complex_cmd = f.readlines()[0]
    #
    # run_dglke(dglke_complex_cmd, exp_dir, "complex_8gpu", overwrite=overwrite, collect_tracing_metrics=collect_tracing_metrics, show_output=show_output)

def fb15k_grid_search(overwrite=False, collect_tracing_metrics=False, show_output=False, short=False):
    exp_dir = "osdi2021/grid_search/fb15k/"
    config = exp_dir + "config.ini"

    grid_config = {"model.scale_factor": [.0001, .001, .01, .1, 1],
                   "training.regularization_coef": [.000001, .00001, .0001, .001, .01, .1, 1],
                   "training.regularization_norm": [1, 2, 3, 4, 5]}

    config_keys = list(grid_config.keys())

    print(grid_config.values())
    for g_config in itertools.product(*grid_config.values()):
        command_line_str = ""
        for i, v in enumerate(g_config):
            command_line_str += "%s=%s " % (config_keys[i], v)
        command_line_str = command_line_str.strip()
        name = command_line_str.replace(" ", "_")
        print(command_line_str)
        run_marius(config, exp_dir, command_line_str.replace(" ", "_"), command_line_str)
        with open(exp_dir + name + "_result.json", 'w') as result_file:
            result = json.load(result_file)
            MRR = result["MRR"][-1]
            print(command_line_str + ": " + MRR)



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
        "staleness_bound": run_staleness_bound,
        "prefetching": run_prefetching,
        "big_embeddings": run_big_embeddings,
        "multi_gpu": run_multi_gpu,
        "fb15k_grid_search": fb15k_grid_search,
        "all": run_all
    }
    parser = argparse.ArgumentParser(description='Reproduce experiments ')
    parser.add_argument('--experiment', metavar='experiment', type=str, choices=experiment_dict.keys(),
                        help='Experiment choices: %(choices)s')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='If true, the results of previously run experiments will be overwritten.')
    parser.add_argument('--collect_tracing_metrics', dest='collect_tracing_metrics', action='store_true',
                        help='If true, dstat and nvidia-smi will collect resource utilization metrics.')
    parser.add_argument('--show_output', dest='show_output', action='store_true',
                        help='If true, the output of each run will be printed directly to the terminal.')
    parser.add_argument('--short', dest='short', action='store_true',
                        help='If true, a shortened version of the experiment will be run')

    args = parser.parse_args()
    experiment_dict.get(args.experiment)(args.overwrite, args.collect_tracing_metrics, args.show_output, args.short)
