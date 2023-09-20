from abc import ABCMeta
from itertools import product, chain
from typing import Union

from numpy import ndarray, array
from pandas import DataFrame

from baddi.datasets.chem import randomize_smiles


class Transformer(metaclass=ABCMeta):
    def __init__(self, data: Union[ndarray, DataFrame], targets: Union[ndarray, DataFrame]):
        self.data = data
        self.targets = targets

    def __transform__(self, molecules: list):
        self.m_map = dict(zip(molecules, list(map(self.transform, molecules))))

    def transform(self, smiles: str):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        print('Data before: ', len(self.data))
        molecules = list(set(chain.from_iterable(self.data)))
        print('Len molecules: ', len(molecules))
        self.__transform__(molecules=molecules)
        print('Len map: ', len(self.m_map))
        tmp_x, tmp_y = [], []
        for (x, y) in zip(self.data, self.targets):
            d1, d2 = x
            m_map_smi1, m_map_smi2 = self.m_map[d1].copy(), self.m_map[d2].copy()
            m_map_smi1.append(d1)
            m_map_smi2.append(d2)
            combs = list(product(m_map_smi1, m_map_smi2))
            tmp_x.extend(combs)
            tmp_y.extend([y for _ in range(len(combs))])
        print('Data end: ', len(tmp_x), len(tmp_y))
        tmp_x, tmp_y = array(tmp_x), array(tmp_y)
        return tmp_x, tmp_y


class Randomizer(Transformer):
    def __init__(self, data: Union[ndarray, DataFrame], targets: Union[ndarray, DataFrame], n_times: int = 1):
        super(Randomizer, self).__init__(data=data, targets=targets)
        self.n_times = n_times

    def transform(self, smiles: str):
        return [smiles] + [randomize_smiles(smiles) for _ in range(self.n_times)]


class ConstrativeTransformer(Randomizer):
    def __init__(self, data: Union[ndarray, DataFrame], targets: Union[ndarray, DataFrame], n_times: int = 1):
        super(ConstrativeTransformer, self).__init__(data=data, targets=targets, n_times=n_times)

    def __call__(self, *args, **kwargs):
        molecules = list(set(self.data))
        self.__transform__(molecules=molecules)
        tmp_x = []
        for x in self.data:
            m_map_x = self.m_map[x].copy()
            combs = list(product([x], m_map_x))
            tmp_x.extend(combs)
        tmp_x = array(tmp_x)
        return tmp_x, None


