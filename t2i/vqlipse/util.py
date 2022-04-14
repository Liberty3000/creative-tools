

def refinement(**args):
    '''
    '''
    import json, pprint, yaml

    if not args['stages']:
        stages = [args]
    else:
        if args['stages'].endswith('.json'):
            with open(args['stages'],'r') as f: _stages = json.load(f)
        elif args['stages'].endswith('.yaml'):
            with open(args['stages'],'r') as f: _stages = yaml.safe_load(f)
        elif isinstance(args['stages'], dict): _stages = args['stages']
        else: _stages = json.loads(args['stages'])
        pprint.pprint(_stages,indent=2)

        stages = []
        for stage in _stages.values():
            dic = {}
            for k,v in args.items():
                dic[k] = v if k not in stage.keys() else stage[k]
            stages.append(dic)
    return stages
