def retrieve_baselines(args, cfgs):
    """
    Retrieve baseline metrics for the given configurations.
    """
    if args.run_name == "six_llava_rank16":
        assert len(cfgs) == 6, "Expected 6 configurations for six_llava_rank16"
        baselines_val = [
            {"TBD": 0.0},
            {"TBD": 0.0},
            {"TBD": 0.0},
            {"TBD": 0.0},
            {"TBD": 0.0},
            {"TBD": 0.0},
        ]
        baselines_test = [
            {"exact_match": 0.6929687499999998},
            {"accuracy": 0.6780512305374183},
            {"relaxed_accuracy": 0.3896},
            {"ANLS": 0.4073408297425932},
            {"CIDEr": 1.3050487678541272},
            {"CIDEr": 0.912880907374865},
        ]
    else:
        print(f"Warning: Baselines for {args.run_name} not found. ")
        return None
    return baselines_val, baselines_test
