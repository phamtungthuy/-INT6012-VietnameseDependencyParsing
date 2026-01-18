class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`.
    This serves as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`Transform` and its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specfic data format.
        data (list[list] or str):
            A list of instances or a filename.
            This will be passed into :meth:`transform.load`.
        kwargs (dict):
            Keyword arguments that will be passed into :meth:`transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`Transform`.
        sentences (list[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
    """

    def __init__(self, transform, data, **kwargs):
        super(Dataset, self).__init__()

        self.transform = transform
        self.sentences = transform.load(data, **kwargs)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        s += ")"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        if not hasattr(self, 'fields'):
            raise RuntimeError("The fields are not numericalized. Please build the dataset first.")
        for d in self.fields.values():
            yield d[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return [getattr(sentence, name) for sentence in self.sentences]

    def __setattr__(self, name, value):
        if 'sentences' in self.__dict__ and name in self.sentences[0]:
            # restore the order of sequences in the buckets
            indices = torch.tensor([i
                                    for bucket in self.buckets.values()
                                    for i in bucket]).argsort()
            for index, sentence in zip(indices, self.sentences):
                setattr(sentence, name, value[index])
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        # only pickle the Transform object and sentences
        return {'transform': self.transform, 'sentences': self.sentences}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def collate_fn(self, batch):
        return {f: d for f, d in zip(self.fields.keys(), zip(*batch))}

    def build(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # numericalize all fields
        self.fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.lengths = [len(i) for i in self.fields[next(iter(self.fields))]]
        self.buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(buckets=self.buckets,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       distributed=distributed),
                                 collate_fn=self.collate_fn)