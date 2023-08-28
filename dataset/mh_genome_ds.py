import os
from dataset import Dataset, Filepath, Label, Dirpath
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Tuple, Set

__ORIG_WD__ = os.getcwd()

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
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags
        }

    #############################
    ### Private Methods Below ###
    #############################
    def _define_private_props(self, 
        coverage: int = 4,
        read_length: int = 128,
        frag_len: int = 128,
        num_frags: int = 256
    ):
        # changable by user
        self._set_coverage(coverage)
        self._set_read_length(read_length)
        self._set_frag_length(frag_len)
        self._set_num_frags(num_frags)
        
        # private props for the use of the class
        self.base_tensor = tf.constant(['A', 'C', 'G', 'T'])
        self._define_dependant_props()


    def _define_dependant_props(self):
        self.dirichlet = tfp.distributions.Dirichlet(concentration=tf.ones([self.num_frags, self.frag_len + self.read_length - 1]))

    
    def _serialize_props(self):
        return {
            "coverage": self.coverage,
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags
        }
    

    def _deserialize_props(self, serialized_obj):
        if "coverage" in serialized_obj:
            self.coverage = serialized_obj["coverage"]
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
        frags_ds = genomes_ds.map(self._extract_minhashed_frags_from_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        sparse_frags_ds = frags_ds.map(self._add_sparsity_to_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return sparse_frags_ds.map(self._encode_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)


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
    def _add_sparsity_to_frags(self, genome, genome_length):
        return tf.where(self._is_frag_base_covered_by_read(genome_length), genome, 'N')


    @tf.function
    def _extract_frags_from_genome(self, genome):
        genome_len = tf.strings.length(genome)
        # assert contig_len >= frag_len, "contig is too short to extract fragments from"
        # Choose a random fragment
        start_indices = tf.random.uniform(
            shape=[self.num_frags],
            minval=0,
            maxval=genome_len - self.frag_len,
            dtype=tf.dtypes.int32
        )
        return tf.strings.bytes_split(tf.strings.substr(genome, start_indices, [self.frag_len] * self.num_frags)).to_tensor(), genome_len

    
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

        return tf.strings.bytes_split(tf.strings.substr(genome, start_indices, [self.frag_len] * self.num_frags)).to_tensor(), genome_len


    @tf.function
    def _encode_frags(self, genome):
        genome = tf.reshape(genome, [self.num_frags, self.frag_len, 1])
        return tf.cast(tf.equal(genome, self.base_tensor), dtype=tf.dtypes.int32)


    @tf.function
    def _is_frag_base_covered_by_read(
        self,
        genome_length
    ):
        num_reads = tf.cast(tf.math.divide_no_nan(tf.math.multiply(
            tf.cast(self.coverage, dtype=tf.float32),
            tf.cast(genome_length, dtype=tf.float32)
        ), tf.cast(self.read_length, tf.float32)), dtype=tf.int32)

        # randomize a vector that adds up to one of size genome_length
        chance_to_be_covered = tf.cast(self.dirichlet.sample(), dtype=tf.float64)
        read_does_not_start_with_base_prob = tf.math.pow(1 - chance_to_be_covered, tf.cast(num_reads, dtype=tf.float64))

        # Build a True/False tensor that states for each base in the fragment, weather a covering read starts with it or not.
        base_has_starting_read = tf.random.uniform(
            # Add read_length - 1 padding at the left of the fragment to cover all bases with identical probability
            [self.num_frags, self.frag_len + self.read_length - 1], 
            minval=0,
            maxval=1,
            dtype=tf.dtypes.float64
        ) >= read_does_not_start_with_base_prob

        # Calculate which bases are covered by the read.
        shift = 1
        base_covered_by_read = base_has_starting_read # 255 bases long
        while(shift < self.read_length):
            # calculate weather the base will be covered by a read.
            rolled = tf.roll(base_covered_by_read, shift=shift, axis=-1) # does not change the shape
            rolled_with_zeros = tf.concat([tf.zeros(rolled.shape[:-1] + min(self.read_length - 1, shift)) == 1, rolled[:, shift:]], axis=-1) #
            base_covered_by_read = rolled_with_zeros | base_covered_by_read
            shift *= 2
        
        # return only the fragment covered-or-not bases
        return base_covered_by_read[:, self.read_length - 1:]
    

    def _set_coverage(self, coverage: float):
        self.coverage = coverage


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