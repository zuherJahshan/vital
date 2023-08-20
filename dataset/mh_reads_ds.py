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
        load: Tuple[bool, Dirpath] = (False, None)
    ):
        self._create_state(mapping, labels, ignore_mapping=load[0])
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


    def get_coverage(self):
        return self.coverage
    

    def set_substitution_rate(self, substitution_rate: float):
        # Except if substitution rate is bigger than 1
        if substitution_rate + self.deletion_rate + self.insertion_rate > 1:
            raise Exception(
                f"""Error rates must be between 0 and 1.
                substitution_rate = {substitution_rate}\tdeletion_rate = {self.deletion_rate}\tinsertion_rate = {self.insertion_rate}"""
            )
        self.substitution_rate = substitution_rate

    def set_insertion_rate(self, insertion_rate: float):
        # Except if insertion rate is bigger than 1
        if insertion_rate + self.deletion_rate + self.substitution_rate > 1:
            raise Exception(
                f"""Error rates must be between 0 and 1.
                insertion_rate = {insertion_rate}\tdeletion_rate = {self.deletion_rate}\tsubstitution_rate = {self.substitution_rate}"""
            )
        self.insertion_rate = insertion_rate

    
    def set_deletion_rate(self, deletion_rate: float):
        # Except if deletion rate is bigger than 1
        if deletion_rate + self.insertion_rate + self.substitution_rate > 1:
            raise Exception(
                f"""Error rates must be between 0 and 1.
                deletion_rate = {deletion_rate}\tinsertion_rate = {self.insertion_rate}\tsubstitution_rate = {self.substitution_rate}"""
            )
        self.deletion_rate = deletion_rate

    
    def get_substitution_rate(self):
        return self.substitution_rate
    

    def get_insertion_rate(self):
        return self.insertion_rate
    

    def get_deletion_rate(self):
        return self.deletion_rate


    def set_read_length(self, length: int):
        # make length a power of two
        length = 2 ** (length - 1).bit_length()
        self.read_length = length


    def set_frag_length(self, length: int):
        # TODO: check validity of length
        self.frag_len = length

    
    def get_frag_length(self):
        return self.frag_len
    

    def get_kmer_length(self):
        return 16


    def set_num_frags(self, num_frags: int):
        # TODO: check validity of num_frags
        self.num_frags = num_frags

    def get_num_frags(self):
        return self.num_frags


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


    def get_examples_filepaths_and_labels(self):
        return self.mapping


    #############################
    ### Private Methods Below ###
    #############################
    def _create_state(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label],
        ignore_mapping: bool
    ):
        self.coverage = 5
        self.substitution_rate = 0
        self.insertion_rate = 0
        self.deletion_rate = 0
        self.read_length = 150
        self.frag_len = 86
        self.num_frags = 400
        self.kmer_length = 16
        self.base_tensor = tf.constant(['A', 'C', 'G', 'T'])
        self.mapping = []
        self.labels = []
        if not ignore_mapping:
            self._add_mapping(mapping, labels)

        self.key = tf.random.stateless_uniform([1], minval=0, maxval=4**self.kmer_length, dtype=tf.int64, seed=[0,0])
            

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
        if "kmer_length" in serialized_obj:
            self.kmer_length = serialized_obj["kmer_length"]
        if not "mapping" in serialized_obj or not "labels" in serialized_obj:
            raise Exception(f"File {modelpath}/dataset.json does not contain mapping or labels.")
        self._add_mapping(serialized_obj["mapping"], serialized_obj["labels"])
        

    def _get_tf_examples_dataset(self):
        accession_files_ds = tf.data.Dataset.from_tensor_slices([accession for accession, _ in self.mapping])
        raw_genomes_ds = accession_files_ds.map(self._process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        genomes_ds = raw_genomes_ds.map(self._clean_raw_genome, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        del_genomes = genomes_ds.map(self._add_deletions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        subs_genomes = del_genomes.map(self._add_substitutions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ins_genomes = subs_genomes.map(self._add_insertions, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        chosen_kmers_ds = ins_genomes.map(self._get_kmers_to_consider, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        minhashed_frags_from_reads_ds = chosen_kmers_ds.map(
            self._extract_minhashed_frags_from_genome,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return minhashed_frags_from_reads_ds.map(self._encode_frags, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    def _get_labels_dataset(self):
        labels_ds = tf.data.Dataset.from_tensor_slices([label for _, label in self.mapping])
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
        genome = tf.strings.regex_replace(genome, '^>', 'N' * self.frag_len)
        return genome
    

    @tf.function
    def _add_substitutions(self, genome):
        if self.substitution_rate == 0:
            return genome

        # Split the genome into bytes
        orig_bases = tf.strings.bytes_split(genome)
        
        # randomize bases from [A,C,G,T] in the size of orig_bases
        random_indices = tf.random.uniform(shape=[tf.strings.length(genome)], minval=0, maxval=4, dtype=tf.int32)
        random_bases = tf.gather(self.base_tensor, random_indices)

        # Randomize substitution events
        substitution_event = tf.random.uniform(shape=[tf.strings.length(genome)], minval=0, maxval=1, dtype=tf.float32) < self.substitution_rate

        # Build new genome
        new_bases = tf.where(substitution_event, random_bases, orig_bases)

        return tf.strings.reduce_join(new_bases)
    

    def _add_insertions(self, genome):
        """
        1. Split the genome into bytes
        2. randomize indel events, use tf.cumsum to update the positions of the genome
        3. use scatter to build new genome of size (genome + number of indel events)
        4. randomize bases for the indel events
        4. use tf.where to build the new genome
        """
        # Split the genome into bytes
        orig_bases = tf.strings.bytes_split(genome)

        # Randomize indel events
        indel_event = tf.random.uniform(shape=[tf.strings.length(genome)], minval=0, maxval=1, dtype=tf.float32) < self.insertion_rate

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



    def _add_deletions(self, genome):
        """
        1. byte split the genome
        2. randomize deletion events
        3. use tf.where to build the new genome
        """

        # Split the genome into bytes
        orig_bases = tf.strings.bytes_split(genome)

        # Randomize deletion events
        deletion_event = tf.random.uniform(shape=[tf.strings.length(genome)], minval=0, maxval=1, dtype=tf.float32) < self.deletion_rate

        # Build new genome
        return tf.strings.reduce_join(tf.where(deletion_event, '', orig_bases))
    

    @tf.function
    def _extract_minhashed_frags_from_genome(self, genome, kmers_to_consider):
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
        for i in range(self.kmer_length):
            kmers += tf.cast(tf.roll(mapped_genome, shift=-1*i, axis=0)[:num_kmers] * tf.pow(5,i), tf.int64)

        # hashed_kmers will be of size of the num_kmers
        # xor hash function appllied as the minhash hashing function
        hashed_kmers = tf.bitwise.bitwise_xor(
            kmers,
            self.key
        )

        modified_hashed_kmers = tf.where(
            kmers_to_consider[:num_kmers],
            hashed_kmers,
            -1 * (2 ** 63)
        )

        frags_to_extract = tf.math.minimum(
            self.num_frags,
            tf.reduce_sum(tf.cast(kmers_to_consider[:num_kmers], dtype=tf.int32))
        )
        
        start_indices = tf.math.top_k(modified_hashed_kmers, k=frags_to_extract).indices

        substr_length = tf.tile(tf.constant([self.frag_len]), [frags_to_extract])
        return tf.strings.bytes_split(tf.strings.substr(genome, start_indices, substr_length)).to_tensor()



    @tf.function
    def _encode_frags(self, genome):
        genome = tf.reshape(genome, [self.num_frags, self.frag_len, 1])
        return tf.cast(tf.equal(genome, self.base_tensor), dtype=tf.dtypes.int32)


    @tf.function
    def _encode_labels(self, label):
        return tf.cast(tf.equal(label, self.labels_tensor), dtype=tf.dtypes.int32)


    @tf.function
    def _get_kmers_to_consider(
        self,
        genome
    ):
        genome_length = tf.strings.length(genome)
        num_reads = tf.cast(tf.math.divide_no_nan(tf.math.multiply(
            tf.cast(self.coverage, dtype=tf.float32),
            tf.cast(genome_length, dtype=tf.float32)
        ), tf.cast(self.read_length, tf.float32)), dtype=tf.int32)

        ## Assume areas from which fragments will be taken must be covered by reads - hence probabilty will change.
        covered_areas = tf.math.minimum(genome_length, self.frag_len* self.num_frags) # O(1)
        read_does_not_start_with_base_prob = tf.math.pow(((covered_areas - 1) / covered_areas), tf.cast(num_reads, dtype=tf.float64))

        # Build a True/False tensor that states for each base in the fragment, weather a covering read starts with it or not.
        base_has_starting_read = tf.random.uniform(
            # Add read_length - 1 padding at the left of the fragment to cover all bases with identical probability
            [genome_length - self.read_length + 1], 
            minval=0,
            maxval=1,
            dtype=tf.dtypes.float64
        ) >= read_does_not_start_with_base_prob

        base_has_starting_read = tf.concat(
            [
                base_has_starting_read,
                tf.zeros(self.read_length - 1, dtype=tf.dtypes.bool),
            ],
            axis=-1
        )

        # Calculate which bases are covered by the read.
        shift = 1
        kmers_to_consider = base_has_starting_read #
        while(shift < self.read_length - self.frag_len):
            # calculate weather the base will be covered by a read.
            rolled = tf.roll(kmers_to_consider, shift=shift, axis=-1) # does not change the shape
            kmers_to_consider = rolled | kmers_to_consider
            shift *= 2
        
        # return only the fragment covered-or-not bases
        return genome, kmers_to_consider

    
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
            "substitution_rate": self.substitution_rate,
            "insertion_rate": self.insertion_rate,
            "deletion_rate": self.deletion_rate,
            "read_length": self.read_length,
            "frag_len": self.frag_len,
            "num_frags": self.num_frags,
            "kmer_length": self.kmer_length,
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