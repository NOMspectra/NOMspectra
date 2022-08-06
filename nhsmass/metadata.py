#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nhsmass. 
#
#    nhsmass is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nhsmass is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nhsmass.  If not, see <http://www.gnu.org/licenses/>.

from typing import Dict, Optional, Mapping
from collections import UserDict


class MetaData(UserDict):
    """
    Class for store and processing metadata of spectrm
    """

    def __init__(self, metadata: Optional[Dict] = None):
        
        """
        metadata: Dict
            Optional. Default None. To add some data into spectrum metedata. Key must be string.
        
        """

        if metadata is None:
            metadata = {}
        elif isinstance(metadata, Mapping) == False:
            raise Exception("Metadata must be dictionary, or None")
        
        metadata = self.make_uniform(metadata)
        
        super().__init__(metadata)

    def make_uniform(self, metadata:Dict):
        
        uniform_metadata = {}
        for key in metadata.keys():
            if isinstance(key, str) == False:
                raise Exception(f"Key in metadata must be string, not {type(key)}")

            new_key = key.lower().replace(" ","_")
            uniform_metadata[new_key] = metadata[key]

        return uniform_metadata

    def add(self, metadata:Mapping):

        if isinstance(metadata, Mapping) == False:
            raise Exception("Metadata must be dictionary, or None")

        if metadata is not None:
            metadata = self.make_uniform(metadata)
        
        #FIXME may be bad solution. It's overwrite existing data
        for key in metadata.keys():
            self[key] = metadata[key]


if __name__ == '__main__':
    pass