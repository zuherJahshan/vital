from typing import Iterator
import os


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