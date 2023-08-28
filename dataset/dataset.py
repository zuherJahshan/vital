from typing import List, Tuple, Set
from abc import ABC, abstractmethod
import random
import os
import tensorflow as tf
import json

Label = str
Filepath = str
Dirpath = str


##################################
############ Dataset #############
##################################

"""
Dataset is an abstract class that will enable you to create, save and load datasets.
To use Dataset you should inherit from it and implement the following methods:
1. _define_private_props(self) -> None:
    This method should add private props to the class.
    It is highly advised not to overwrite the private props of the Dataset class.
    The private props are defined in the **create_state** method.

2. _serialize_props(self) -> dict:
    This method should return a dictionary containing the serialized private props of the class.

3. _deserialize_props(self, obj, serialized_obj) -> None:
    This method should deserialize the private props from a json object.

4. update_props(self, props: dict):
    This method should update the private props of the class according to the props dictionary.
    The private props are defined in the define_private_props method.
    This method is called when loading a dataset.

5. _get_tf_examples_dataset(self, repeats: int = None, shuffle_buffer_size: int = None, batch_size: int = 32):
    This method should return a tf.data.Dataset containing the pre-processed examples.
"""


class Dataset(object):
    def __init__(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label] = None,
        load: Tuple[bool, Dirpath] = (False, None)
    ) -> None:
        self._create_state(mapping, labels, ignore_mapping=load[0])
        if load[0]:
            self._load_state(load[1])


    def get_labels(self) -> List[Label]:
        return self.labels
    

    def get_batch_size(self) -> int:
        return self.batch_size
    

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size


    def get_size(self):
        return len(self.mapping)
    

    def save(self, dirpath: Dirpath) -> None:
        # Check that you are not overwriting an existing model
        os.makedirs(dirpath, exist_ok=True)
        
        serialized_obj = self._serialize()
        with open(f"{dirpath}/dataset.json", 'w') as f:
            json.dump(serialized_obj, f)

        with open(f"{dirpath}/config.txt", 'w') as f:
            # write the type of the class
            f.write(f"{type(self).__name__}\n")


    def get_tf_dataset(
        self,
        repeats: int = None,
        no_labels: bool = False,
        shuffle_buffer_size: int = None,
    ):
        if no_labels:
            return self._get_tf_examples_dataset(repeats, shuffle_buffer_size, self.batch_size)
        else :
            tf_dataset = tf.data.Dataset.zip((self._get_tf_examples_dataset(), self._get_labels_dataset()))
        if repeats == None:
            tf_dataset = tf_dataset.repeat()
        else:
            tf_dataset = tf_dataset.repeat(repeats)
        if shuffle_buffer_size:
            tf_dataset = tf_dataset.shuffle(shuffle_buffer_size)
        tf_dataset = tf_dataset.ragged_batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset
    

    def get_examples_filepaths_and_labels(self):
        return self.mapping


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


    @abstractmethod
    def update_props(self, props: dict):
        pass


    @abstractmethod
    def get_props(self) -> dict:
        pass

    ########################################
    ############ Private Methods ###########
    ########################################

    def _create_state(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label] = None,
        ignore_mapping: bool = False
    ) -> None:
        self._define_private_props()
        self.mapping = []
        self.labels = []
        self.labels_tensor = None
        if not ignore_mapping:
            self._add_mapping(mapping, labels)


    def _load_state(self, dirpath: Dirpath) -> None:
        # load the json from the modelpath
        
        # check that appropriate files exist
        if not os.path.exists(f"{dirpath}/dataset.json"):
            raise Exception(f"File {dirpath}/dataset.json does not exist.")
        
        # load the json
        with open(f"{dirpath}/dataset.json", 'r') as f:
            serialized_obj = json.load(f)

        self._deserialize_props(serialized_obj)

        # check that the json contains the mapping and labels, and deserialize them
        if not "mapping" in serialized_obj or not "labels" in serialized_obj:
            raise Exception(f"File {dirpath}/dataset.json does not contain mapping or labels.")
        self._add_mapping(serialized_obj["mapping"], serialized_obj["labels"])

    
    def _add_mapping(
        self,
        mapping: List[Tuple[Filepath, Label]],
        labels: List[Label] = None
    ) -> None:
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


    def _check_mapping_validity(self, mapping):
        for filepath, label in mapping:
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist.")


    def _get_labels_tensor(self) -> tf.Tensor:
        return tf.constant(self.labels)


    def _serialize(self) -> dict:
        serialized_obj = self._serialize_props()
        serialized_obj["mapping"] = self.mapping
        serialized_obj["labels"] = self.labels
        return serialized_obj


    def _get_labels_dataset(self):
        labels_ds = tf.data.Dataset.from_tensor_slices([label for _, label in self.mapping])
        labels_ds = labels_ds.map(self._encode_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return labels_ds


    @tf.function
    def _encode_labels(self, label):
        return tf.cast(tf.equal(label, self.labels_tensor), dtype=tf.dtypes.int32)


    @tf.function
    def _process_path(self, file_path):
        return tf.io.read_file(file_path)


    @abstractmethod
    def _define_private_props(self) -> None:
        pass


    @abstractmethod
    def _serialize_props(self) -> dict:
        pass


    @abstractmethod
    def _get_tf_examples_dataset(self, repeats: int = None, shuffle_buffer_size: int = None, batch_size: int = 32):
        pass


    @abstractmethod
    def _deserialize_props(self, serialized_obj) -> None:
        pass