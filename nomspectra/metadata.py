#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nomspectra. 
#
#    nomspectra is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomspectra is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomspectra.  If not, see <http://www.gnu.org/licenses/>.

from typing import Dict, Optional, Mapping
from collections import UserDict


class MetaData(UserDict):
    """
    Class for store and processing metadata of spectrum
    """

    def __init__(self, metadata: Optional[Dict] = None):
        """
        Parameters
        ----------
        metadata: Dict
            Optional. Default None. To add some data into spectrum metedata. Key must be string.
        """

        if metadata is None:
            metadata = {}
        elif isinstance(metadata, Mapping) == False:
            raise Exception("Metadata must be dictionary, or None")
        
        metadata = self._make_uniform(metadata)
        
        super().__init__(metadata)

    def _make_uniform(self, metadata:Dict) -> Dict:
        """
        Uniform metadata keys

        Parameters
        ----------
        metadata: Dict
            Dictonary for uniform

        Return
        ------
        Dict
        """
        
        uniform_metadata = {}
        for key in metadata.keys():
            if isinstance(key, str) == False:
                raise Exception(f"Key in metadata must be string, not {type(key)}")

            new_key = key.lower().replace(" ","_")
            uniform_metadata[new_key] = metadata[key]

        return uniform_metadata

    def add(self, metadata:Mapping) -> None:
        """
        add new fields to metadata dictonary

        Parameters
        ----------
        metadata: Mapping
            new fields for adding. For example {'operator':'Alex'}
        """

        if isinstance(metadata, Mapping) == False:
            raise Exception("Metadata must be dictionary, or None")

        if metadata is not None:
            metadata = self._make_uniform(metadata)

        for key in metadata.keys():
            self[key] = metadata[key]
    
    @staticmethod
    def combine_two_name(spec1, spec2) -> str:
        """
        combine two names from metadata into 
        one string with '_' as separator

        Parameters
        ----------
        spec1: Spectrum
            first spectrum object
        spec2: Spectrum
            second spectrum object

        Return
        ------
        str
        """
        
        name1 = name2 = '_'
        if 'name' in spec1.metadata:
            name1 = spec1.metadata['name']
        if 'name' in spec2.metadata:
            name2 = spec2.metadata['name']
        return f"{name1}_{name2}"

if __name__ == '__main__':
    pass