def get_info_preprocess(functions, pipelines):
    info = {}
    for idx, func in enumerate(functions):
        func_dict = func.to_dict()
        module = func_dict['module']
        name = func_dict['name']
        parameters = func_dict['parameters']
        info[f'func_{idx}'] = str({'name': f'{module}.{name}', 'parameters': parameters})

    for idx, pipe in enumerate(pipelines):
        pipe_dict = pipe.to_dict()
        module = pipe_dict['module']
        name = pipe_dict['name']
        functions_name = pipe_dict['functions']
        parameters = pipe_dict['parameters']
        info[f'pipeline_{idx}'] = str({'name': f'{module}.{name}', 'functions': functions_name, 'parameters': parameters})

    return info