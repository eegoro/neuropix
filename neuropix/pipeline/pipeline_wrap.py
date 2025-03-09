class PipelineWrap:
    def __init__(self, module, pipeline_name, functions=None, **kwargs):
        self.module = module
        self.pipeline_name = pipeline_name
        self.pipeline_func = getattr(module, pipeline_name)
        self.functions = functions or []
        self.params = kwargs
    
    def __call__(self, data, time_array, fs):
        return self.pipeline_func(
            data, 
            time_array, 
            fs, 
            functions=self.functions, 
            **self.params
        )
    
    def to_dict(self):
        functions_name = []
        for func in self.functions:
            functions_name.append(func.func_name)
            
        return {
            "name": self.pipeline_name,
            "module": self.module.__name__.split('.')[-1],
            "functions": functions_name,
            "parameters": self.params
        }