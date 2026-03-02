from importlib import import_module


_EXPORTS = {
    "evaluate": (".evaluation", "evaluate"),
    "evaluate_fc": (".evaluation_fc", "evaluate_fc"),
    "finetune": (".finetune", "finetune"),
    "finetune_fc": (".finetune_fc", "finetune_fc"),
    "evaluate_wise_ft": (".wiseft", "evaluate_wise_ft"),
    "finetune_icarl": (".icarl", "iCaRL"),
    "eval_single_image": (".evaluation", "eval_single_image"),
    "custom_finetune": (".training", "custom_finetune"),
    "ProbeLayer": (".probes", "ProbeLayer"),
    "EncoderProbes": (".probes", "EncoderProbes"),
    "smoke_test": (".smoke_test", "smoke_test"),
    "test": (".test", None),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
