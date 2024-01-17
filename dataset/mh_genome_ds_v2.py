import os
from dataset import Dataset, Filepath, Label, Dirpath
import tensorflow as tf
# import tensorflow_probability as tfp
from typing import List, Tuple, Set


__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../utils")

import utils

os.chdir(__ORIG_WD__)

Contig = tf.Tensor


class MHGenomeDS(Dataset):
    def __init__(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label] = None,
        load: Tuple[bool, Dirpath] = (False, None)
    ):
        super().__init__(mapping, labels, load)


    def update_props(self, props: dict):
        if "coverage" in props:
            self._set_coverage(props["coverage"])
        if "substitution_rate" in props:
            self._set_substitution_rate(props["substitution_rate"])
        if "insertion_rate" in props:
            self._set_insertion_rate(props["insertion_rate"])
        if "deletion_rate" in props:
            self._set_deletion_rate(props["deletion_rate"])
        if "read_length" in props:
            self._set_read_length(props["read_length"])
        if "frag_len" in props:
            self._set_frag_length(props["frag_len"])
        if "num_frags" in props:
            self._set_num_frags(props["num_frags"])

        self._define_dependant_props()


    def get_props(self) -> dict:
        return {
            "coverage": self.coverage,
            "substitution_rate": self.substitution_rate,
            "insertion_rate": self.insertion_rate,
            "deletion_rate": self.deletion_rate,
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags
        }

    #############################
    ### Private Methods Below ###
    #############################
    def _define_private_props(self, 
        coverage: int = 4,
        substitution_rate: float = 0,
        insertion_rate: float = 0,
        deletion_rate: float = 0,
        read_length: int = 128,
        frag_len: int = 128,
        num_frags: int = 256
    ):
        # changable by user
        self._set_coverage(coverage)
        self._set_substitution_rate(substitution_rate)
        self._set_insertion_rate(insertion_rate)
        self._set_deletion_rate(deletion_rate)
        self._set_read_length(read_length)
        self._set_frag_length(frag_len)
        self._set_num_frags(num_frags)
        
        # private props for the use of the class
        self.base_tensor = tf.constant(['A', 'C', 'G', 'T'])
        self._define_dependant_props()


    def _define_dependant_props(self):
        pass
        # self.dirichlet = tfp.distributions.Dirichlet(concentration=tf.ones([self.num_frags, self.frag_len + self.read_length - 1]))

    
    def _serialize_props(self):
        return {
            "coverage": self.coverage,
            "substitution_rate": self.substitution_rate,
            "insertion_rate": self.insertion_rate,
            "deletion_rate": self.deletion_rate,
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags
        }
    

    def _deserialize_props(self, serialized_obj):
        if "coverage" in serialized_obj:
            self.coverage = serialized_obj["coverage"]
        if "substitution_rate" in serialized_obj:
            self.substitution_rate = serialized_obj["substitution_rate"]
        if "insertion_rate" in serialized_obj:
            self.insertion_rate = serialized_obj["insertion_rate"]
        if "deletion_rate" in serialized_obj:
            self.deletion_rate = serialized_obj["deletion_rate"]
        if "read_length" in serialized_obj:
            self.read_length = serialized_obj["read_length"]
        if "frag_len" in serialized_obj:
            self.frag_len = serialized_obj["frag_len"]
        if "num_frags" in serialized_obj:
            self.num_frags = serialized_obj["num_frags"]

    
    def _get_tf_examples_dataset(self):
        accession_files_ds = tf.data.Dataset.from_tensor_slices([accession for accession, _ in self.mapping])
        raw_genomes_ds = accession_files_ds.map(self._process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        genomes_ds = raw_genomes_ds.map(self._clean_raw_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        genomes_with_coverage_ds = genomes_ds.map(self._add_coverage_to_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        sparse_genomes_ds = genomes_with_coverage_ds.map(self._add_sparsity, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        del_genomes_ds = sparse_genomes_ds.map(self._add_deletions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        subs_genomes_ds = del_genomes_ds.map(self._add_substitutions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ins_genomes_ds = subs_genomes_ds.map(self._add_insertions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        frags_ds = ins_genomes_ds.map(self._extract_minhashed_frags_from_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return frags_ds.map(self._encode_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    @tf.function
    def _add_coverage_to_genome(self, genome):
        genome_length = tf.strings.length(genome)
        coverage_per_base = self._compute_coverage_per_base(genome_length)
        return genome, coverage_per_base


    @tf.function
    def _compute_coverage_per_base(self, genome_length):
        num_reads = tf.cast(tf.math.ceil(self.coverage * genome_length / self.read_length), tf.int32)
        read_starts_per_base = tf.concat(
            [
                utils.random_partition(
                    balls = num_reads,
                    bins = genome_length - self.read_length + 1,
                ),
                tf.zeros(self.read_length, dtype=tf.int32)
            ],
            axis=-1
        )
        read_ends_per_base = -1*tf.roll(read_starts_per_base, self.read_length, axis=0)
        coverage_per_base = tf.cumsum(read_starts_per_base + read_ends_per_base)
        return coverage_per_base[:-1]


    @tf.function
    def _clean_raw_genome(self, raw_genome):
        # Remove descriptor lines
        genome = tf.strings.regex_replace(raw_genome, '>.*\n', '>')
        # Remove newlines
        genome = tf.strings.regex_replace(genome, '\n', '')
        # Remove empty lines
        genome = tf.strings.regex_replace(genome, '^>', 'N' * self.frag_len)
        return genome


    @tf.function
    def _add_sparsity(self, genome, coverage_per_base):
        split_genome = tf.strings.bytes_split(genome)
        return tf.strings.reduce_join(tf.where(coverage_per_base > 0, split_genome, 'N')), coverage_per_base


    @tf.function
    def _add_deletions(self, genome, coverage_per_base):
        if self.deletion_rate == 0:
            return genome, coverage_per_base
        
        split_genome = tf.strings.bytes_split(genome)

        # Compute number of deletions
        deletion_event = tf.random.uniform(
            shape=[tf.strings.length(genome)],
            minval=0,
            maxval=1,
            dtype=tf.dtypes.float32
        ) < tf.where(
                coverage_per_base > 0,
                tf.math.pow(self.deletion_rate, tf.cast(tf.math.ceil(coverage_per_base / 2), tf.float32)),
                0
            )

        coverage_per_base = tf.boolean_mask(coverage_per_base, tf.logical_not(deletion_event))

        return tf.strings.reduce_join(tf.where(deletion_event, '', split_genome)), coverage_per_base


    @tf.function
    def _add_substitutions(self, genome, coverage_per_base):
        if self.substitution_rate == 0:
            return genome, coverage_per_base

        # Split the genome into bytes
        split_genome = tf.strings.bytes_split(genome)
        
        # randomize bases from [A,C,G,T] in the size of orig_bases
        random_indices = tf.random.uniform(shape=[tf.strings.length(genome)], minval=0, maxval=4, dtype=tf.int32)
        random_bases = tf.gather(self.base_tensor, random_indices)

        # Randomize substitution events
        substitution_event = tf.random.uniform(
            shape=[tf.strings.length(genome)],
            minval=0,
            maxval=1,
            dtype=tf.float32
        ) < tf.where(
                coverage_per_base > 0,
                tf.math.pow(self.substitution_rate, tf.cast(tf.math.ceil(coverage_per_base / 2), tf.float32)),
                0
            )

        # Build new genome
        new_bases = tf.where(substitution_event, random_bases, split_genome)

        return tf.strings.reduce_join(new_bases), coverage_per_base


    @tf.function
    def _add_insertions(self, genome, coverage_per_base):
        """
        1. Split the genome into bytes
        2. randomize indel events, use tf.cumsum to update the positions of the genome
        3. use scatter to build new genome of size (genome + number of indel events)
        4. randomize bases for the indel events
        4. use tf.where to build the new genome
        """
        if self.insertion_rate == 0:
            return genome

        # Split the genome into bytes
        orig_bases = tf.strings.bytes_split(genome)

        # Randomize indel events
        indel_event = tf.random.uniform(
            shape=[tf.strings.length(genome)],
            minval=0,
            maxval=1,
            dtype=tf.float32
        ) < tf.where(
                coverage_per_base > 0,
                tf.math.pow(self.insertion_rate, tf.cast(tf.math.ceil(coverage_per_base / 2), tf.float32)),
                0
            )

        # Update the positions of the genome
        cumsum = tf.cumsum(tf.cast(indel_event, tf.int32))
        positions = cumsum + tf.range(tf.strings.length(genome))

        # Build new genome
        scattered_orig_bases = tf.scatter_nd(tf.expand_dims(positions, axis=1), orig_bases, [tf.strings.length(genome) + cumsum[-1]])

        # Randomize bases for the indel events
        random_indices = tf.random.uniform(
            shape=[tf.strings.length(genome) + cumsum[-1]],
            minval=0,
            maxval=4,
            dtype=tf.int32
        )
        random_bases = tf.gather(self.base_tensor, random_indices)

        # update indel event dimenstions
        indel_event = tf.concat([indel_event, tf.zeros([cumsum[-1]], dtype=tf.bool)], axis=0)

        return tf.strings.reduce_join(tf.where(indel_event, random_bases, scattered_orig_bases))


    @tf.function
    def _extract_frags_from_genome(self, genome):
        genome_len = tf.strings.length(genome)[0]
        # assert contig_len >= frag_len, "contig is too short to extract fragments from"
        # Choose a random fragment
        start_indices = tf.random.uniform(
            shape=[self.num_frags],
            minval=0,
            maxval=genome_len - self.frag_len,
            dtype=tf.dtypes.int32
        )
        
        return tf.strings.bytes_split(tf.strings.substr(genome, start_indices, [self.frag_len] * self.num_frags)).to_tensor()

    
    @tf.function
    def _extract_minhashed_frags_from_genome(self, genome):
        kmer_size = 16
        genome_len = tf.strings.length(genome)

        # define hash table
        keys = tf.constant(["A", "C", "G", "T"])
        values = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        default_value = tf.constant(0, dtype=tf.int32)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)

        # Define the hash table
        table = tf.lookup.StaticHashTable(initializer, default_value)

        # Look up the values
        mapped_genome = table.lookup(tf.strings.bytes_split(genome))

        # Extract kmers
        num_kmers = genome_len - self.frag_len + 1
        kmers = tf.zeros([num_kmers], dtype = tf.int64)
        for i in range(kmer_size):
            kmers += tf.cast(tf.roll(mapped_genome, shift=-1*i, axis=0)[:num_kmers] * tf.pow(5,i), tf.int64)

        # Extract minhashes
        hashed_kmers = tf.bitwise.bitwise_xor(
            kmers,
            tf.random.stateless_uniform([1], minval=0, maxval=4**kmer_size, dtype=tf.int64, seed=[0,0])
        )
        
        start_indices = tf.math.top_k(hashed_kmers, k=self.num_frags).indices

        return tf.strings.bytes_split(tf.strings.substr(genome, start_indices, [self.frag_len] * self.num_frags)).to_tensor()


    @tf.function
    def _encode_frags(self, genome):
        genome = tf.reshape(genome, [self.num_frags, self.frag_len, 1])
        return tf.cast(tf.equal(genome, self.base_tensor), dtype=tf.dtypes.int32)
    

    def _set_coverage(self, coverage: float):
        self.coverage = coverage


    def _set_substitution_rate(self, substitution_rate: float):
        self.substitution_rate = substitution_rate


    def _set_deletion_rate(self, deletion_rate: float):
        self.deletion_rate = deletion_rate


    def _set_insertion_rate(self, insertion_rate: float):
        self.insertion_rate = insertion_rate


    def _set_read_length(self, length: int):
        # make length a power of two
        length = 2 ** (length - 1).bit_length()
        self.read_length = length


    def _set_frag_length(self, length: int):
        # TODO: check validity of length
        self.frag_len = length

   
    def _set_num_frags(self, num_frags: int):
        # TODO: check validity of num_frags
        self.num_frags = num_frags