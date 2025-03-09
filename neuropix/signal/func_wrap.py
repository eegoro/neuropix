class FuncWrap:
    def __init__(self, module, func_name, **kwargs):
        self.module = module
        self.func_name = func_name
        self.params = kwargs
        self.func = getattr(module, func_name)(**kwargs)
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
        
    def to_dict(self):
        return {
            "name": self.func_name,
            "module": self.module.__name__.split('.')[-1],
            "parameters": self.params
        }