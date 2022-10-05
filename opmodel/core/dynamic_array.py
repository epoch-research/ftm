## Dynamically resizing Numpy array.

## Downloaded from https://github.com/dylanwal/numpy_dynamic_array (and modified)
##
## License:
##
##   Copyright (c) 2021 Dylan Walsh
##   All rights reserved.
##   
##   Redistribution and use in source and binary forms, with or without
##   modification, are permitted provided that the following conditions are
##   met:
##   
##       * Redistributions of source code must retain the above copyright
##          notice, this list of conditions and the following disclaimer.
##   
##       * Redistributions in binary form must reproduce the above
##          copyright notice, this list of conditions and the following
##          disclaimer in the documentation and/or other materials provided
##          with the distribution.
##   
##       * Neither the name of the numpy_dynamic_array Developers nor the names of any
##          contributors may be used to endorse or promote products derived
##          from this software without specific prior written permission.
##   
##   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
##   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
##   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
##   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
##   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
##   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
##   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
##   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
##   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
##   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
##   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Union, Any

import numpy as np

number_type = (int, float)
array_type = (np.ndarray, list, tuple)
value_type = array_type + number_type
value_alias = Union[int, float, list[Any], tuple[Any], np.ndarray]
index_alias = Union[int, slice, tuple[int]]


class DynamicArray:
    """
    A class to dynamically grow numpy array as data is added in an efficient manner.
    For arrays or column vectors (new rows added, but no new added columns).

    Attributes:
    ----------
    shape: int, array_type
        Starting shape of the dynamic array
    index_expansion: bool
        allow setting indexing outside current capacity
        will set all values between previous size to new value to zero

    Example
    -------
    a = DynamicArray((100, 2))
    a.append(np.ones((20, 2)))
    a.append(np.ones((120, 2)))
    a.append(np.ones((10020, 2)))
    print(a.data)
    print(a.data.shape)
    """

    def __init__(self, shape: Union[int, tuple[int], list[int]] = 100, dtype=None, index_expansion: bool = True, data: np.ndarray = None):
        self._data = np.zeros(shape, dtype) if dtype is not None else np.zeros(shape)
        self.capacity = self._data.shape[0]
        self.size = self._data.shape[0]
        self.index_expansion = index_expansion

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__().replace("array", f'DynamicArray(size={self.size}, capacity={self.capacity})')

    def __getitem__(self, index: index_alias):
        return self.data[index]

    def __setitem__(self, index: index_alias, value: value_alias):
        max_index = self._get_max_index(index)
        if not self.index_expansion:
            if max_index > self.size:
                raise IndexError(f"Attempting to reach index outside of data array. "
                                 f"Size: {self.size}, attempt index: {max_index}\n"
                                 f"If you want the array to grow with indexing, set index_expansion to True.")

        self._capacity_check_index(max_index)

        # add data
        if isinstance(index, int) and index < 0 or \
                isinstance(index, slice) and (index.start < 0 or index.stop < 0) or \
                isinstance(index, tuple) and any(i < 0 for i in index if isinstance(i, int)):
            # handle negative indexing
            self.data[index] = value
        else:
            # handling positive indexing
            self._data[index] = value

        # update capacity and size (if it was outside current size)
        if max_index > self.size:
            capacity_change = max_index - self.size
            self.capacity -= capacity_change
            self.size += capacity_change

    @staticmethod
    def _get_max_index(index: index_alias) -> int:
        """ get max index """
        if isinstance(index, slice):
            return int(index.stop)
        if isinstance(index, tuple):
            return int(index[0]) + 1

        # must be an int
        return index + 1

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
        except AttributeError:
            # check numpy for function call
            attr = object.__getattribute__(self.data, name)

        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result
            return newfunc

        else:
            return attr

    def __add__(self, a):
        return self.data + a

    def __eq__(self, a):
        if a.__class__ is self.__class__ or a.__class__ in array_type:
            return np.equal(self.data, a)

    def __gt__(self, a):
        return self.data > a

    def __ge__(self, a):
        return self.data >= a

    def __lt__(self, a):
        return self.data < a

    def __le__(self, a):
        return self.data <= a

    def __floordiv__(self, a):
        return self.data // a

    def __mod__(self, a):
        return self.data % a

    def __mul__(self, a):
        return self.data * a

    def __neg__(self):
        return -self.data

    def __pow__(self, a):
        return self.data ** a

    def __truediv__(self, a):
        return self.data / a

    def __sub__(self, a):
        return self.data - a

    def __len__(self):
        return self.size

    def append(self, x: value_alias):
        """ Add data to array. """
        add_size = self._capacity_check(x)

        # Add new data to array
        self._data[self.size:self.size + add_size] = x
        self.size += add_size
        self.capacity -= add_size

    def _capacity_check_index(self, index: int = 0):
        if index > len(self._data):
            add_size = (index-len(self._data)) + self.capacity
            self.grow_capacity(add_size)

    def _capacity_check(self, x: value_alias):
        """ Check if there is room for the new data. """
        if isinstance(x, number_type):
            add_size = 1
        elif isinstance(x, array_type):
            add_size = len(x)
        else:
            raise ValueError("Invalid item to add.")

        if add_size > self.capacity:
            self.grow_capacity(add_size)

        return add_size

    def grow_capacity(self, add_size: int):
        """ Grows the capacity of the _data array. """
        # calculate what change is needed.
        change_need = add_size - self.capacity

        # make new larger data array
        shape_ = list(self._data.shape)
        if shape_[0] + self.capacity > add_size:
            # double in size
            self.capacity += shape_[0]
            shape_[0] = shape_[0] * 2
        else:
            # if doubling is not enough, grow to fit incoming data exactly.
            self.capacity += change_need
            shape_[0] = shape_[0] + change_need
        newdata = np.zeros(shape_, dtype=self._data.dtype)

        # copy data into new array and replace old one
        newdata[:self._data.shape[0]] = self._data
        self._data = newdata

    def grow_size(self, add_size: int):
        """ Grows the size of the array. """
        new_size = self.size + add_size
        if new_size > self.capacity:
          self.grow_capacity(new_size - self.capacity)
        self.size = new_size

    @property
    def data(self):
        """ Returns data without extra spaces. """
        return self._data[:self.size]


    # -------------------------------------------------------------------------
    # Serialization

    def __getstate__(self):
        state = {
            '_data': self._data,
            'size': self.size,
            'capacity': self.capacity,
        }
        return state

    def __setstate__(self, state):
        self._data = state['_data']
        self.size = state['size']
        self.capacity = state['capacity']

    def __reduce__(self):
        return (self.__class__, (self.shape, self.dtype, self.index_expansion), self.__getstate__())
