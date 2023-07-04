import tensorflow as tf
import os
from typing import Iterator, List, Tuple
import json
import random

__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../utils/")
from utils import *

os.chdir(__ORIG_WD__)

Contig = tf.Tensor
Label = str
Filepath = str
Dirpath = str
Label = str


class Dataset(object):
    def __init__(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label] = None,
        minhash_dataset: bool = False,
        load: Tuple[bool, Dirpath] = (False, None)
    ):
        self._create_state(mapping, labels, minhash_dataset, ignore_mapping=load[0])
        if load[0]:
            self._load_state(load[1])

    
    def get_labels(self):
        return self.labels


    def get_size(self):
        return len(self.mapping)
    

    def get_number_of_labels(self):
        return self.labels_tensor.shape[0]


    def save(self, modelpath: Dirpath):
        # Check that you are not overwriting an existing model
        os.makedirs(modelpath, exist_ok=True)
        
        serialized_obj = self._serialize()
        with open(f"{modelpath}/dataset.json", 'w') as f:
            json.dump(serialized_obj, f)


    def get_tf_dataset(self,
                       repeats: int = None,
                       shuffle_buffer_size: int = None,
                       batch_size: int = 32,):
        tf_dataset = tf.data.Dataset.zip((self._get_tf_examples_dataset(), self._get_labels_dataset()))
        if repeats == None:
            tf_dataset = tf_dataset.repeat()
        else:
            tf_dataset = tf_dataset.repeat(repeats)
        if shuffle_buffer_size:
            tf_dataset = tf_dataset.shuffle(shuffle_buffer_size)
        tf_dataset = tf_dataset.batch(batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset
        

    def set_coverage(self, coverage: float):
        self.coverage = coverage
        print("mrow")


    def get_coverage(self):
        return self.coverage


    def set_read_length(self, length: int):
        # make length a power of two
        length = 2 ** (length - 1).bit_length()
        self.read_length = length


    def set_frag_length(self, length: int):
        # TODO: check validity of length
        self.frag_len = length

    
    def get_frag_length(self):
        return self.frag_len


    def set_num_frags(self, num_frags: int):
        # TODO: check validity of num_frags
        self.num_frags = num_frags


    def existing_examples(self, examples: List[Tuple[Filepath, Label]]):
        # checks if the following examples exist in the dataset
        existing_examples = []
        dataset_examples = set([filepath for filepath, _ in self.mapping])
        for example, _ in examples:
            if example in dataset_examples:
                existing_examples.append(example)
        return existing_examples
    

    def add_examples(self, examples: List[Tuple[Filepath, Label]]):
        if len(self.existing_examples(examples)) > 0:
            raise Exception("Some examples already exist in the dataset.")
        self._add_mapping(examples, None)


    #############################
    ### Private Methods Below ###
    #############################
    def _create_state(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label],
        minhash_dataset: bool,
        ignore_mapping: bool
    ):
        self.coverage = 5
        self.read_length = 128
        self.frag_len = 128
        self.num_frags = 256
        self.base_tensor = tf.constant(['A', 'C', 'G', 'T'])
        self.minhash_dataset = minhash_dataset
        self.mapping = []
        self.labels = []
        if not ignore_mapping:
            self._add_mapping(mapping, labels)
            

    def _add_mapping(self, mapping: List[Tuple[Filepath, Label]], labels: List[Label]):
        
        # update the mappings
        self._check_mapping_validity(mapping)
        random.shuffle(mapping)
        self.mapping: List[Tuple[Filepath, Label]] = mapping
        
        # update the labels
        existing_labels_set = set(self.labels)
        if not labels:
            new_labels_set = set([label for _, label in mapping])
        else:
            new_labels_set = set(labels)
        existing_labels_set.update(new_labels_set)
        self.labels = list(existing_labels_set)
        self.labels.sort()

        # update the labels tensor
        self.labels_tensor = self._get_labels_tensor()
            

    def _load_state(self, modelpath):
        # load the jsonj from the modelpath
        
        # check that appropriate files exist
        if not os.path.exists(f"{modelpath}/dataset.json"):
            raise Exception(f"File {modelpath}/dataset.json does not exist.")
        
        # load the json
        with open(f"{modelpath}/dataset.json", 'r') as f:
            serialized_obj = json.load(f)

        # set the state
        if "coverage" in serialized_obj:
            self.coverage = serialized_obj["coverage"]
        if "read_length" in serialized_obj:
            self.read_length = serialized_obj["read_length"]
        if "frag_len" in serialized_obj:
            self.frag_len = serialized_obj["frag_len"]
        if "num_frags" in serialized_obj:
            self.num_frags = serialized_obj["num_frags"]
        if "minhash_dataset" in serialized_obj:
            self.minhash_dataset = serialized_obj["minhash_dataset"]       
        if not "mapping" in serialized_obj or not "labels" in serialized_obj:
            raise Exception(f"File {modelpath}/dataset.json does not contain mapping or labels.")
        self._add_mapping(serialized_obj["mapping"], serialized_obj["labels"])
        

    def _get_tf_examples_dataset(self):
        accession_files_ds = tf.data.Dataset.from_tensor_slices([accession for accession, label in self.mapping])
        raw_genomes_ds = accession_files_ds.map(self._process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        genomes_ds = raw_genomes_ds.map(self._clean_raw_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if not self.minhash_dataset:
            frags_ds = genomes_ds.map(self._extract_frags_from_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            frags_ds = genomes_ds.map(self._extract_minhashed_frags_from_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        sparse_frags_ds = frags_ds.map(self._add_sparsity_to_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return sparse_frags_ds.map(self._encode_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    def _get_labels_dataset(self):
        labels_ds = tf.data.Dataset.from_tensor_slices([label for accession, label in self.mapping])
        labels_ds = labels_ds.map(self._encode_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return labels_ds


    @tf.function
    def _process_path(self, file_path):
        return tf.io.read_file(file_path)


    @tf.function
    def _clean_raw_genome(self, raw_genome):
        # Remove descriptor lines
        genome = tf.strings.regex_replace(raw_genome, '>.*\n', '>')
        # Remove newlines
        genome = tf.strings.regex_replace(genome, '\n', '')
        # Remove empty lines
        genome = tf.strings.regex_replace(genome, '^>', '')
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
        # window_size = tf.cast(tf.math.divide_no_nan(
        #     tf.cast(num_kmers, tf.float32), 
        #     tf.cast(self.num_frags, tf.float32)),
        # tf.int32)
        # reshaped_kmers = tf.reshape(kmers[:self.num_frags*window_size], [self.num_frags, window_size])
        # hashed_kmers = tf.strings.to_hash_bucket_fast(tf.strings.as_string(reshaped_kmers), num_buckets=2**32)
        # start_indices = tf.argmax(hashed_kmers, axis=1, output_type=tf.dtypes.int32) + (tf.range(self.num_frags, dtype=tf.int32) * window_size)
        # xor kmers with uniform random salt
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
    def _encode_labels(self, label):
        return tf.cast(tf.equal(label, self.labels_tensor), dtype=tf.dtypes.int32)


    @tf.function
    def _is_frag_base_covered_by_read(
        self,
        genome_length
    ):
        num_reads = tf.cast(tf.math.divide_no_nan(tf.math.multiply(
            tf.cast(self.coverage, dtype=tf.float32),
            tf.cast(genome_length, dtype=tf.float32)
        ), tf.cast(self.read_length, tf.float32)), dtype=tf.int32)

        ## Assume areas from which fragments will be taken must be covered by reads - hence probabilty will change.
        print(genome_length)
        covered_areas = tf.math.minimum(genome_length, self.frag_len* self.num_frags) # O(1)
        read_does_not_start_with_base_prob = tf.math.pow(((covered_areas - 1) / covered_areas), tf.cast(num_reads, dtype=tf.float64))

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


    def _sparse_frags_generator(self):
        #iterate over all provided files
        for fasta_filepath, label in self.mapping:
            genome, genome_length = self._extract_genome_from_file(fasta_filepath)
            frags = self._extract_frags_from_genome(genome, genome_length) # THE BUG IS HERE
            # frags = self._generate_random_frags()
            frag_base_covered_by_read = self._is_frag_base_covered_by_read(genome_length)
            yield self._encode_frags(tf.where(frag_base_covered_by_read, frags, 0)), self._get_label_one_hot(label)

    
    def _get_labels_tensor(self):
        return tf.constant(self.labels)


    def _check_mapping_validity(self, mapping):
        for filepath, label in mapping:
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist.")
            if not is_fasta_file(filepath):
                raise Exception(f"File {filepath} is not a fasta file.")


    def _serialize(self):
        return {
            "coverage": self.coverage,
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags,
            "minhash_dataset": self.minhash_dataset,
            "mapping": self.mapping,
            "labels": self.labels
        }


def load_dataset(modelpath: Dirpath) -> Dataset:
    if os.path.exists(f"{modelpath}/dataset.json"):
        return Dataset(mapping = None, load=(True, modelpath))
    else:
        raise Exception(f"File {modelpath}/dataset.json does not exist.")


def remove_dataset(modelpath: Dirpath) -> None:
    if os.path.exists(f"{modelpath}/dataset.json"):
        os.remove(f"{modelpath}/dataset.json")