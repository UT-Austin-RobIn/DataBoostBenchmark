class AttrDict(dict):
    '''AttrDict wraps around a python dictionary and makes keys accessible
    as attributes: dictionary.key instead of dictionary["key"].
    '''
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, dictionary):
        self = dictionary