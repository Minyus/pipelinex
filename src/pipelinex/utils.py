def dict_of_list_to_list_of_dict(dict_of_list):
    return [
        dict(zip(dict_of_list.keys(), vals)) for vals in zip(*dict_of_list.values())
    ]


def list_of_dict_to_dict_of_list(list_of_dict):
    return {k: [d[k] for d in list_of_dict] for k in list_of_dict[0]}
