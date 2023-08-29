from typing import Iterator
import os
import tensorflow as tf


def sub_dirpath_of_dirpath(dirpath: str, sub_dirpath: str) -> bool:
    return os.path.abspath(sub_dirpath).startswith(os.path.abspath(dirpath))


def is_fasta_file(filename):
    return filename.endswith('.fa') or filename.endswith('.fasta') or filename.endswith('.fna')

def is_fastq_file(filename):
    return filename.endswith('.fq') or filename.endswith('.fastq')


def get_fasta_files_in_dir(fasta_dirname) -> Iterator[str]:
        for dirpath, _, filenames in os.walk(fasta_dirname):
            for filename in filenames:
                if is_fasta_file(filename):
                    yield os.path.join(dirpath, filename)


# create a function that prints progress bar
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


############################
##### Tensorflow utils #####
############################
@tf.function
def random_partition(balls, bins):
    """
    1 represents a ball
    0 represents a partition
    """
    # randomize the ones indexes
    balls_indexes = tf.random.shuffle(tf.range(balls+bins-1))[:balls]
    balls_and_partitions = tf.concat([
        tf.scatter_nd(
            indices=tf.expand_dims(balls_indexes, 1),
            updates=tf.ones(balls, dtype=tf.dtypes.int32),
            shape=[bins+balls-1]
        ), 
        [0]
    ], axis=-1)
    
    # cumsum of balls and partitions
    cumsum = tf.cumsum(balls_and_partitions)

    summed_balls_per_partition = tf.boolean_mask(cumsum, balls_and_partitions == 0)

    prev_balls_per_partition = tf.concat([[0], summed_balls_per_partition[:-1]], axis = -1)

    return summed_balls_per_partition - prev_balls_per_partition