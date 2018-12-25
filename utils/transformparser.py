from torchvision import transforms
from typing import List, Dict, Any
import inspect


class TransformParser(object):
    def __init__(self, net_params=None):
        if net_params is None:
            net_params = dict()
        self.net_params = net_params

    def _parse_single_param(self, param: Any) -> Any:
        # string, eval it
        if isinstance(param, str):
            return eval(param, self.net_params)
        # list of params
        elif isinstance(param, list):
            return [self._parse_single_param(p) for p in param]
            # ps = []
            # for pa in param:
            #     ps.append(self._parse_single_param(pa))
            # return ps
        # dict, parse as a transform
        elif isinstance(param, dict):
            return self._parse_single_transform(param)[0]
        else:
            return param

    def _parse_single_transform(self, single: dict):
        tmodule = transforms if 'module' not in single else __import__(
            single['module'], fromlist=[single['name']])
        topt = getattr(tmodule, single['name'])

        params: dict = single['params']
        for k, v in params.items():
            # if isinstance(v, str) and v in self.net_params:
            #     params[k] = self.net_params[v]
            params[k] = self._parse_single_param(v)

        if 'multi' in single:
            multi = single['multi']
        else:
            multi = (tmodule != transforms) and inspect.isfunction(topt)

        return topt(**params), multi

    def parse(self, translist: List[dict]):
        ts = []
        for trans in translist:
            t, multi = self._parse_single_transform(trans)

            if multi:
                ts.extend(t)
            else:
                ts.append(t)

        return transforms.Compose(ts)
