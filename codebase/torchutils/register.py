'''
Modification from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/utils/registry.py

Under the License Apache License 2.0
License text can be found at https://github.com/open-mmlab/mmdetection/blob/master/LICENSE
'''

TYPE = "type_"


class Register(dict):
    def __init__(self, name):
        self["__name"] = name

    @property
    def name(self):
        return self["__name"]

    def register(self, module):
        if not callable(module):
            raise TypeError('module must be callable, but got {}'.format(type(module)))
        module_name = module.__name__
        if module_name in self:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self[module_name] = module

    def build_from(self, cfg: dict, default_args=None):
        if TYPE not in cfg:
            raise ValueError(f"There is no type item {self.name} in the cfg.")
        args = cfg.copy()
        obj_type = args.pop(TYPE)
        if isinstance(obj_type, str):
            obj_type = self.get(obj_type)
            if obj_type is None:
                raise KeyError('{} is not in the {} registry'.format(
                    obj_type, self.name))
        else:
            raise ValueError(f"The key {obj_type} must be a string.")
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        return obj_type(**args)
