import os
import tensorflow as tf
import numpy as np
import glob

class ProdReadsDS(object):
    def __init__(
        self,
        kmer_len,
        frag_len,
        num_frags
    ):
        self.key = tf.random.stateless_uniform([1], minval=0, maxval=4**kmer_len, dtype=tf.int64, seed=[0,0]).numpy()[0]
        self.kmer_len = kmer_len
        self.frag_len = frag_len
        self.num_frags = num_frags
        self.base_tensor = tf.constant(['A', 'C', 'G', 'T'])
        self.base_to_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}


    def get_tf_dataset(
        self,
        dirpath: str
    ) -> tf.data.Dataset:
        frags_dataset = tf.data.Dataset.from_generator(
            generator = lambda: self.frags_generator(
                dirpath,
            ),
            output_signature=(
                tf.TensorSpec(shape=(400), dtype=tf.dtypes.string),
                tf.TensorSpec(shape=(), dtype=tf.dtypes.string)
            ),
        )
        frags_dataset = frags_dataset.map(
            self._encode_frags,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return frags_dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)


    def frags_generator(
        self,
        dirpath: str
    ) -> tf.Tensor:
        # extract fastq files from dirpath
        reads_file_pathes = glob.glob(os.path.join(dirpath, '*.fq')) + glob.glob(os.path.join(dirpath, '*.fastq'))
        for reads_file_path in reads_file_pathes:
            frags = self._get_frags(reads_file_path)
            yield frags, reads_file_path


    def _get_frags(
        self,
        fastq_file
    ):
        reads = self._get_reads(fastq_file)
        frags = []
        for read_idx, read in enumerate(reads):
            for frag_idx in range(len(read) - self.frag_len + 1):
                frags.append(((read_idx, frag_idx), self._get_signed_frag_value(read[frag_idx:frag_idx + self.frag_len])))
        # sort and filter out same values
        frags.sort(key=lambda x: x[1], reverse=True)
        frags = frags[0:0] + [frags[i] for i in range(1, len(frags)) if frags[i][1] != frags[i-1][1]]
        return [reads[read_idx][frag_idx:frag_idx + self.frag_len] for (read_idx, frag_idx), _ in frags[:self.num_frags]]


    def _get_reads(
        self,
        fastq_file
    ):
        """Reads a fastq file and returns a list of reads."""
        reads = []
        with open(fastq_file) as f:
            while True:
                f.readline() # ignore the first line
                seq = f.readline().strip()
                f.readline() # ignore the third line
                qual = f.readline().strip()
                if len(seq) == 0:
                    break
                reads.append(seq)
        return reads


    def _get_signed_frag_value(
        self,
        frag
    ):
        kmer = 0
        for i in range(self.kmer_len):
            kmer += np.multiply(self.base_to_int.get(frag[i], 0), np.power(5, i, dtype=np.int32), dtype=np.int32)

        # perform bitwise XOR with key
        kmer ^= self.key
        return kmer


    @tf.function
    def _encode_frags(
        self,
        genome_frags,
        reads_filepath
    ):
        genome_frags = tf.strings.bytes_split(genome_frags)
        genome_frags = tf.reshape(genome_frags, [self.num_frags, self.frag_len, 1])
        return tf.cast(tf.equal(genome_frags, self.base_tensor), dtype=tf.dtypes.int32), reads_filepath
    

