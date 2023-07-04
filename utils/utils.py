from typing import Iterator
import os


def sub_dirpath_of_dirpath(dirpath: str, sub_dirpath: str) -> bool:
    return os.path.abspath(sub_dirpath).startswith(os.path.abspath(dirpath))


def is_fasta_file(filename):
    return filename.endswith('.fa') or filename.endswith('.fasta') or filename.endswith('.fna')


def get_fasta_files_in_dir(fasta_dirname) -> Iterator[str]:
        for dirpath, _, filenames in os.walk(fasta_dirname):
            for filename in filenames:
                if is_fasta_file(filename):
                    yield os.path.join(dirpath, filename)