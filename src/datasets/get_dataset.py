from .amass import AMASS

def get_dataset(name="amass"):
    return AMASS


def get_datasets(parameters, split="train"):
    DATA = AMASS

    if split == 'all':
        train = DATA(split='train', **parameters)
        test = DATA(split='vald', **parameters)

        # add specific parameters from the dataset loading
        train.update_parameters(parameters)
        test.split = 'vald'
        test.update_parameters(parameters)
    else:
        dataset = DATA(split=split, **parameters)
        train = dataset

        # test: shallow copy (share the memory) but set the other indices
        from copy import copy
        test = copy(train)
        test.split = test

        # add specific parameters from the dataset loading
        dataset.update_parameters(parameters)

    datasets = {"train": train,
                "test": test}

    return datasets
