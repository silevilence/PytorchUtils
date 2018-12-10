from torchvision import transforms
from typing import List, Dict


class TransformParser(object):
    def __init__(self, net_params=None):
        if net_params is None:
            net_params = dict()
        self.net_params = net_params

    def _parse_single(self, single: dict):
        tmodule = transforms if 'module' not in single else __import__(
            single['module'], fromlist=[single['name']])
        topt = getattr(tmodule, single['name'])

        params: dict = single['params']
        for k, v in params.items():
            if isinstance(v, str) and v in self.net_params:
                params[k] = self.net_params[v]

        if 'multi' in single:
            multi = single['multi']
        else:
            multi = (tmodule != transforms)

        return topt(**params), multi

    def parse(self, translist: List[dict]):
        ts = []
        for trans in translist:
            t, multi = self._parse_single(trans)

            if multi:
                ts.extend(t)
            else:
                ts.append(t)
        
        return transforms.Compose(ts)
