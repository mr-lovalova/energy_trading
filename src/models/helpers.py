def get_num_input_output(data_loader):
    dataloader_iter = iter(data_loader)
    sample_batch = next(dataloader_iter)
    *ex_features, ex_target = sample_batch
    num_features = [feature.shape[1] for feature in ex_features]
    num_target = ex_target.shape[1]
    return num_features, num_target
