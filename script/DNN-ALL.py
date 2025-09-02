# encoding:utf-8
def train(dictionary_filename, data_filename, id_varname, target_varname, output_dir, sam_method='RandomOverSampler',
          cv_folds=5, hidden_units_list=[[20, 20], [50, 50]], lr=1e-4, batch_size=32, epochs=300, patience=2,
          pearson_cutoff=1.1, seed=1):
    ## import modules
    import os
    import logging
    import pickle
    import uuid
    import ctypes
    import pandas as pd
    import numpy as np
    import random
    import datetime
    import warnings
    import scipy.stats as stats
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shutil
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path
    from retrying import retry
    from filelock import FileLock, Timeout
    from sklearn import model_selection
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, \
        precision_recall_curve, f1_score
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.over_sampling import RandomOverSampler, RandomOverSampler, SMOTE
    from imblearn.combine import SMOTETomek

    # set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")  # Disable GPU usage
    # the below configs that work for pytorch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    run_id = uuid.uuid4().hex[:6]
    if os.name == 'nt' and not ctypes.windll.shell32.IsUserAnAdmin():
        raise PermissionError("This script requires administrator privileges on Windows. Please run as Administrator.")
    ## Write log
    # create output and output train directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)
    if not os.path.exists('Train'):
        os.makedirs('Train')
    output_train_dir = os.path.join(output_dir, 'Train')
    # do basic configuration for the logging system
    logging.root.handlers = []  # Clear existing handlers
    filename_trainlog = os.path.join('Train', 'Train.txt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=filename_trainlog,
        filemode='w',
        force=True)
    ## Import and parse the data sets
    logging.info("Begin import and parse the data sets...")
    datetime_beg = datetime.datetime.now()
    # import dict
    dict_use = pd.read_csv(dictionary_filename, encoding='utf-8', dtype={
        'table_name': str,
        'table_biz': str,
        'var_name': str,
        'var_type': str,
        'var_biz': str})
    vars_dict_use = set(dict_use.var_name)
    vars_df = set(pd.read_csv(data_filename, nrows=0).columns)
    vars = vars_dict_use & vars_df
    dict_use = dict_use[dict_use['var_name'].isin(vars)]
    vars_fea = vars - set(id_varname + [target_varname])
    vars_fea_num = set(dict_use[dict_use['var_type'] == 'numerical']['var_name']) & vars_fea
    vars_fea_cha = vars_fea - vars_fea_num
    vars_list = list(vars)

    def get_var_type(var):
        if dict_use[dict_use['var_name'] == var]['var_type'].values[0] == 'numerical':
            return np.float32
        else:
            return str

    var_name_list = list(dict_use['var_name'])
    var_biz_list = list(dict_use['var_biz'])
    table_name_list = list(dict_use['table_name'])
    table_biz_list = list(dict_use['table_biz'])
    var_name_list.append('intercept')
    var_biz_list.append('intercept')
    table_name_list.append('intercept')
    table_biz_list.append('intercept')
    # import sample
    col_dtypes = [get_var_type(var) for var in vars_list]
    col_dtypes_dict_use = {vars_list[i]: col_dtypes[i] for i in np.arange(len(col_dtypes))}
    train = pd.read_csv(data_filename, usecols=vars_list, dtype=col_dtypes_dict_use, encoding='utf-8')
    # print sample data info
    logging.info('overall state before preprocess:')
    logging.info('sample number: ' + str(train.shape[0]))
    logging.info('variable number :' + str(len(vars_fea)))
    logging.info('begin data preprocessing...')
    # parse data
    # delete duplicated samples
    train = train.drop_duplicates()
    logging.info('There have been ' + str(train.duplicated().sum()) + ' duplicated samples removed.')

    # Ensure consistent data types and handle missing values
    y_train = np.array([val.strip() for val in train[target_varname]]).copy().astype(np.int32)
    vars_fea_num_list = sorted(list(vars_fea_num), key=None, reverse=False)
    vars_fea_cha_list = sorted(list(vars_fea_cha), key=None, reverse=False)

    # Handle numerical features
    X_train_num = train[vars_fea_num_list].copy()
    X_train_fac = train[vars_fea_cha_list].copy()

    # Handle categorical features
    for var in vars_fea_cha_list:
        X_train_fac.loc[:, var] = [val.strip() for val in X_train_fac.loc[:, var].astype(str)]

    # Replace blank with 'other'
    X_train_fac = X_train_fac.fillna('other')

    # Remove variables with only one value except null
    vars_fac_one_val_list = []
    for var in vars_fac_one_val_list:
        if len(set(X_train_fac[var]) - set([np.NaN])) == 1:
            vars_fac_one_val_list.append(var)
            vars_fea_cha = vars_fea_cha - set(vars_fac_one_val_list)
            vars_fea_cha_list = list(vars_fea_cha)
            logging.info('There have been ' + str(
                len(vars_fac_one_val_list)) + ' factor variable with only one value except nan removed.')
            logging.info('They are ' + str(vars_fac_one_val_list))
    else:
        logging.info("There is no factor variable with only one value except nan.")

    # Save values with samples more than 500
    vals_fac_value_pairs = {}
    for var in X_train_fac.columns:
        se_tmp = X_train_fac[[var]].groupby(var).size()
        values = np.array(se_tmp.index[se_tmp > 500])
        vals_fac_value_pairs[var] = values
        X_train_fac.loc[~ X_train_fac[var].isin(values), var] = 'other'

    vars_fea_cha_list_ori = vars_fea_cha_list.copy()

    # Dummy encoding
    for var in vars_fea_cha_list:
        dummies = pd.get_dummies(X_train_fac[var]).rename(columns=lambda x: var + '__' + str(x))
        X_train_fac = pd.concat([X_train_fac, dummies], axis=1)
        X_train_fac.drop([var], inplace=True, axis=1)

    vars_fea_cha_list = np.array(X_train_fac.columns)
    # numeric preprocess
    # imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train_num = imp.fit_transform(X_train_num)

    # Standardize
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)

    vars_fea_num_list_ori = vars_fea_num_list.copy()

    # Merge data
    X_train_ori = np.hstack((X_train_fac.values, X_train_num))
    vars_fea_list = []
    vars_fea_list.extend(vars_fea_cha_list)
    vars_fea_list.extend(vars_fea_num_list_ori)
    vars_fea_list_ori = []
    vars_fea_list_ori.extend(vars_fea_cha_list_ori)
    vars_fea_list_ori.extend(vars_fea_num_list_ori)
    # feature selectiong
    # Removing features with low variance
    threshold = (.95 * (1 - .95))
    sel = VarianceThreshold(threshold=threshold)
    X_train = sel.fit_transform(X_train_ori)
    inds_colsleft = np.where(sel.variances_ >= threshold)[0]

    # Correlation check
    X_corr = np.corrcoef(X_train_ori, rowvar=0)
    vec_tar = y_train
    inds_remove = list()
    for i in inds_colsleft:
        vec_i = X_train_ori[:, i]
        p_i = stats.pearsonr(vec_i, vec_tar)[0]
        for j in np.array(inds_colsleft)[np.array(inds_colsleft) > i]:
            if abs(X_corr[i, j]) >= pearson_cutoff:
                vec_j = X_train_ori[:, j]
                p_j = stats.pearsonr(vec_j, vec_tar)[0]
                if abs(p_i) >= abs(p_j):
                    inds_remove.append(j)
                else:
                    inds_remove.append(i)
                    break

    inds_colsleft = list(set(inds_colsleft) - set(inds_remove))
    vars_fea_list = list(np.array(vars_fea_list)[inds_colsleft])
    X_train = X_train_ori[:, inds_colsleft]

    ## sample balancing 
    if sam_method == 'RandomUnderSampler':
        sam = RandomUnderSampler(random_state=seed)
    elif sam_method == 'TomekLinks':
        sam = TomekLinks(sampling_strategy='not minority', random_state=seed)
    elif sam_method == 'RandomOverSampler':
        sam = RandomOverSampler(sampling_strategy='minority', random_state=seed)
    elif sam_method == 'SMOTE':
        sam = SMOTE(sampling_strategy='minority', random_state=seed)
    elif sam_method == 'SMOTETomek':
        sam = SMOTETomek(sampling_strategy='not minority', random_state=seed)
    else:
        print("Sorry! There is no such sampling method!")
    X_train_resampled, y_train_resampled = sam.fit_resample(X_train, y_train)
    logging.info('The sample method was ' + sam_method + '.')
    logging.info('There have been ' + str(X_train.shape[0]) + ' samples before resampling.')
    logging.info('including ' + str(y_train.sum()) + ' positive samples.')
    logging.info('and ' + str(X_train.shape[0] - y_train.sum()) + ' negative samples.')
    logging.info('After resampling ' + str(X_train.shape[0] - X_train_resampled.shape[0]) + ' samples was removed.')
    logging.info('There have been ' + str(y_train_resampled.shape[0]) + ' samples remained after resampling.')
    logging.info('including ' + str(y_train_resampled.sum()) + ' positive samples.')
    logging.info('and ' + str(X_train_resampled.shape[0] - y_train_resampled.sum()) + ' negative samples.')

    # save preprocess file
    num_classes = 2
    col_dtypes_dict_use = {var: col_dtypes_dict_use[var] for var in vars_fea_list_ori + id_varname + [target_varname]}
    datetime_end = datetime.datetime.now()
    datetime_intv = datetime_end - datetime_beg
    logging.info("End import and parse the data sets! time taken: {}".format(datetime_intv))
    ## Create feature columns to describe the data
    X_train = pd.DataFrame(X_train_resampled, columns=vars_fea_list).values
    y_train = pd.Series(y_train_resampled, dtype=np.int32).values
    input_size = X_train.shape[1]
    # define function for model
    logging.info("Begin train the model...")
    datetime_beg = datetime.datetime.now()

    ## define dataset and dataloader
    class DNNDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    def collate_fn(batch):
        features, labels = zip(*batch)
        features = torch.stack(features)
        labels = torch.stack(labels)
        return features, labels

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    ## define model
    class DNNModel(nn.Module):
        def __init__(self, input_size, hidden_units):
            super(DNNModel, self).__init__()
            self.dnn_layers = []
            for i, units in enumerate(hidden_units):
                self.dnn_layers.append(nn.Linear(input_size if i == 0 else hidden_units[i - 1], units))
                self.dnn_layers.append(nn.LayerNorm(units))
                self.dnn_layers.append(nn.LeakyReLU(negative_slope=0.05))
                if i == 0:
                    self.dnn_layers.append(nn.Dropout(0.1))
                else:
                    self.dnn_layers.append(nn.Dropout(0.5))
            self.dnn_layers.append(nn.Linear(hidden_units[-1], 1))
            self.dnn_layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*self.dnn_layers)

        def forward(self, x):
            return self.model(x)

    ## define BCELossWithL1
    class BCELossWithL1(nn.Module):
        def __init__(self, model, lambda_l1=5e-3):
            super().__init__()
            self.model = model
            self.lambda_l1 = lambda_l1
            self.bce_loss = nn.BCELoss()

        def forward(self, y_pred, y_true):
            loss = self.bce_loss(y_pred, y_true)

            l1_reg = torch.tensor(0., device=y_pred.device)
            for param in self.model.parameters():
                l1_reg += torch.sum(torch.abs(param))

            total_loss = loss + self.lambda_l1 * l1_reg
            return total_loss

    def find_optimal_threshold(y_true, y_scores):
        """Find optimal threshold using precision-recall curve and F1 score"""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def create_writer_safe(log_dir):
        lock_path = Path(log_dir).parent / "lock"
        try:
            with FileLock(lock_path, timeout=1):
                if os.path.exists(log_dir):
                    for f in Path(log_dir).glob('events.out*'):
                        f.unlink()
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir, flush_secs=1)
                return writer
        except Timeout:
            raise Exception(f"File lock timeout in {log_dir}")

            # Set directory permissions
            if os.name == 'nt':  # Windows
                import win32security
                import ntsecuritycon as con
                import win32file
                import win32con

                # Get current user's SID
                username = os.getlogin()
                domain = os.environ['USERDOMAIN']
                user_sid, _, _ = win32security.LookupAccountName(domain, username)

                # Create a new DACL
                security = win32security.SECURITY_DESCRIPTOR()
                dacl = win32security.ACL()

                # Add full control for the current user
                dacl.AddAccessAllowedAce(
                    win32security.ACL_REVISION,
                    con.FILE_ALL_ACCESS,
                    user_sid
                )

                # Add full control for Administrators group
                admin_sid = win32security.LookupAccountName(None, "Administrators")[0]
                dacl.AddAccessAllowedAce(
                    win32security.ACL_REVISION,
                    con.FILE_ALL_ACCESS,
                    admin_sid
                )

                security.SetSecurityDescriptorDacl(1, dacl, 0)
                win32security.SetFileSecurity(
                    log_dir,
                    win32security.DACL_SECURITY_INFORMATION,
                    security
                )

                # Set directory attributes
                win32file.SetFileAttributes(log_dir, win32con.FILE_ATTRIBUTE_NORMAL)

                # Set permissions for all files in the directory
                for root, dirs, files in os.walk(log_dir):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        win32security.SetFileSecurity(
                            dir_path,
                            win32security.DACL_SECURITY_INFORMATION,
                            security
                        )
                    for f in files:
                        file_path = os.path.join(root, f)
                        win32security.SetFileSecurity(
                            file_path,
                            win32security.DACL_SECURITY_INFORMATION,
                            security
                        )
            else:  # Unix/Linux
                os.chmod(log_dir, 0o777)

            # Create writer with shorter flush interval
            writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

            # Test write permissions
            try:
                writer.add_scalar('test/permission', 0, 0)
                writer.flush()
            except Exception as e:
                logging.error(f"Failed to write test log: {str(e)}")
                raise

            return writer
        except Exception as e:
            logging.error(f"Failed to create writer for {log_dir}: {str(e)}")
            raise

    def safe_write_tensorboard(writer, metrics, epoch):
        """Safely write metrics to TensorBoard with error handling"""
        try:
            for metric_name, value in metrics.items():
                writer.add_scalar(metric_name, value, epoch)
            writer.flush()

            # Set file permissions after writing
            if os.name == 'nt':  # Windows
                import win32security
                import ntsecuritycon as con
                import win32file
                import win32con

                # Get current user's SID
                username = os.getlogin()
                domain = os.environ['USERDOMAIN']
                user_sid, _, _ = win32security.LookupAccountName(domain, username)

                # Create a new DACL
                security = win32security.SECURITY_DESCRIPTOR()
                dacl = win32security.ACL()

                # Add full control for the current user
                dacl.AddAccessAllowedAce(
                    win32security.ACL_REVISION,
                    con.FILE_ALL_ACCESS,
                    user_sid
                )

                # Add full control for Administrators group
                admin_sid = win32security.LookupAccountName(None, "Administrators")[0]
                dacl.AddAccessAllowedAce(
                    win32security.ACL_REVISION,
                    con.FILE_ALL_ACCESS,
                    admin_sid
                )

                security.SetSecurityDescriptorDacl(1, dacl, 0)

                # Set permissions for the log file
                log_file = os.path.join(writer.log_dir, f"events.out.tfevents.*")
                for f in Path(writer.log_dir).glob(log_file):
                    try:
                        win32security.SetFileSecurity(
                            str(f),
                            win32security.DACL_SECURITY_INFORMATION,
                            security
                        )
                    except Exception as e:
                        logging.warning(f"Failed to set permissions for {f}: {str(e)}")
        except Exception as e:
            logging.warning(f"Failed to write TensorBoard logs: {str(e)}")
            # Don't raise the exception, just log it and continue

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    def safe_save_model(model, path):
        torch.save(model.state_dict(), path)

    # Create dataset and dataloader
    generator_train = torch.Generator()
    generator_train.manual_seed(seed)
    dataset_train = DNNDataset(X_train, y_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        generator=generator_train,
        num_workers=0)
    # For testing, use the same data
    generator_test = torch.Generator()
    generator_test.manual_seed(seed)
    dataset_test = DNNDataset(X_train, y_train)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        generator=generator_test,
        num_workers=0)
    # k fold data
    kfold = model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    ## find the best parameter for hidden_depth
    auc_best = 0
    model_tmp_dir = os.path.join('Train', 'dnn_tmp_model')
    if os.path.exists(model_tmp_dir):  # Clean tmp model directory
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir)
    iter_paras = 0
    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=[
        'Train Type',
        'Hidden Units',
        'Epochs',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'AUC'
    ])
    for hidden_units in hidden_units_list:
        iter_paras += 1
        tmp_dir = os.path.join(model_tmp_dir, str(iter_paras))
        os.makedirs(tmp_dir, exist_ok=True)

        # Define the model, loss function, and optimizer
        model = DNNModel(input_size, hidden_units).to(device)
        model.apply(weights_init)
        criterion = BCELossWithL1(model, lambda_l1=5e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07, amsgrad=False,
                               weight_decay=1e-5)

        # Ensure the directory exists
        log_dir = os.path.join(tmp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        loss_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        auc_list = []
        epoch_list = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
            # Ensure the directory exists
            fold_log_dir = os.path.join(log_dir, f"{fold + 1}_{run_id}")
            try:
                # Cleanup before creating directory
                if os.path.exists(fold_log_dir):
                    for f in Path(fold_log_dir).glob('events.out*'):
                        try:
                            f.unlink()
                        except Exception as e:
                            print(f"Cleanup warning: {str(e)}")
                # Create directory with explicit permissions
                os.makedirs(fold_log_dir, exist_ok=True)
                # Windows specific permission setting
                if os.name == 'nt':
                    import win32security
                    import win32con
                    sd = win32security.GetFileSecurity(fold_log_dir, win32security.DACL_SECURITY_INFORMATION)
                    user, _, _ = win32security.LookupAccountName("", os.getlogin())
                    dacl = win32security.ACL()
                    dacl.AddAccessAllowedAce(win32con.ACL_REVISION, win32con.GENERIC_ALL, user)
                    sd.SetSecurityDescriptorDacl(1, dacl, 0)
                    win32security.SetFileSecurity(fold_log_dir, win32security.DACL_SECURITY_INFORMATION, sd)

            except Exception as e:
                print(f"Directory handling error: {str(e)}")
                continue
            try:
                # Create writer with explicit flush_secs
                writer = create_writer_safe(fold_log_dir)
                # Create datasets for the current fold
                dataset_train_folder = DNNDataset(X_train[train_idx], y_train[train_idx])
                dataset_test_folder = DNNDataset(X_train[test_idx], y_train[test_idx])
                # Create DataLoaders for the current fold
                generator_train_folder = torch.Generator()
                generator_train_folder.manual_seed(seed)
                dataloader_train_folder = DataLoader(
                    dataset_train_folder,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                    generator=generator_train_folder,
                    num_workers=0)
                generator_test_folder = torch.Generator()
                generator_test_folder.manual_seed(seed)
                dataloader_test_folder = DataLoader(
                    dataset_test_folder,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    shuffle=False,
                    generator=generator_test_folder,
                    num_workers=0)

                early_stop_counter = 0
                best_auc = 0
                for epoch in range(epochs):
                    # Train
                    model.train()
                    train_loss = 0.0
                    for batch in dataloader_train_folder:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    # Calculate average training loss
                    train_loss /= len(dataloader_train_folder)
                    # Validate
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        all_labels = []
                        all_outputs = []
                        for batch in dataloader_test_folder:
                            X_batch, y_batch = batch
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            test_outputs = model(X_batch)
                            test_loss = criterion(test_outputs.squeeze(), y_batch)
                            val_loss += test_loss.item()
                            # Store outputs and labels for global AUC calculation
                            all_labels.extend(y_batch.cpu().numpy())
                            all_outputs.extend(test_outputs.squeeze().cpu().numpy())

                        val_loss /= len(dataloader_test_folder)

                        y_true = np.array(all_labels)
                        y_scores = np.array(all_outputs)

                        optimal_threshold = find_optimal_threshold(y_true, y_scores)

                        y_pred = (y_scores > optimal_threshold).astype(int)

                        if len(np.unique(y_true)) > 1:
                            val_auc = roc_auc_score(y_true, y_scores)
                        else:
                            warnings.warn("Only one class present in validation set. ROC AUC score is not defined.")
                            val_auc = 0.5

                        val_accuracy = accuracy_score(y_true, y_pred)
                        val_precision = precision_score(y_true, y_pred, zero_division=0)
                        val_recall = recall_score(y_true, y_pred, zero_division=0)
                        val_f1 = f1_score(y_true, y_pred, zero_division=0)

                        metrics = {
                            'Loss/Train': train_loss,
                            'Loss/Validate': val_loss,
                            'Accuracy/Validate': val_accuracy,
                            'Precision/Validate': val_precision,
                            'Recall/Validate': val_recall,
                            'F1/Validate': val_f1,
                            'AUC/Validate': val_auc,
                            'Threshold/Validate': optimal_threshold
                        }
                        safe_write_tensorboard(writer, metrics, epoch + 1)
                        # Print metrics
                        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, "
                                     f"Val Accuracy: {val_accuracy}, Val Precision: {val_precision}, "
                                     f"Val Recall: {val_recall}, Val F1: {val_f1}, Val AUC: {val_auc}, "
                                     f"Optimal Threshold: {optimal_threshold}")
                        # Early stopping
                        if (val_auc > best_auc) and (val_auc > 0.5) and (val_recall > 0.02) and (val_recall < 1):
                            best_auc = val_auc
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if (early_stop_counter >= patience) and (val_auc > 0.5):
                                logging.info(f"Early stopping at epoch {epoch + 1}")
                                break  # Exit epoch loop
            finally:
                try:
                    if 'writer' in locals():
                        writer.flush()
                        writer.close()
                        if hasattr(writer, '_get_file_writer'):
                            writer._get_file_writer().close()
                        import gc
                        gc.collect()
                except Exception as e:
                    logging.warning(f"Writer closure warning: {str(e)}")
            loss_list.append(val_loss)
            accuracy_list.append(val_accuracy)
            precision_list.append(val_precision)
            recall_list.append(val_recall)
            f1_list.append(val_f1)
            auc_list.append(val_auc)
            epoch_list.append(epoch + 1)

        # Evaluate the model's effectiveness
        loss_iter = np.mean(loss_list)
        accuracy_iter = np.mean(accuracy_list)
        precision_iter = np.mean(precision_list)
        recall_iter = np.mean(recall_list)
        f1_iter = np.mean(f1_list)
        auc_iter = np.mean(auc_list)
        # In metrics calculation:
        epoch_list = [e for e in epoch_list if not np.isnan(e)]
        epoch_iter = round(np.mean(epoch_list)) if epoch_list else 0
        # Create new row for dataframe
        new_row = {
            'Train Type': '5-fold cross validation',
            'Hidden Units': str(hidden_units),
            'Epochs': epoch_iter,
            'Accuracy': accuracy_iter,
            'Precision': precision_iter,
            'Recall': recall_iter,
            'F1 Score': f1_iter,
            'AUC': auc_iter
        }
        # Append to dataframe
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
        if auc_iter >= auc_best:
            hidden_units_best = hidden_units
            epoch_best = epoch_iter
            accuracy_best = accuracy_iter
            precision_best = precision_iter
            recall_best = recall_iter
            f1_best = f1_iter
            auc_best = auc_iter
            best_model = model
        logging.info(
            "\nHidden Units: {0}\nEpochs: {1}\nAccuracy: {2}\nPrecision: {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
                str(hidden_units), epoch_iter, accuracy_iter, precision_iter, recall_iter, f1_iter, auc_iter))

    # print the best results
    logging.info(
        "\ntrain best parameters:\nHidden Units: {0}\nEpochs: {1}\nAccuracy: {2}\nPrecision: {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
            str(hidden_units_best), epoch_best, accuracy_best, precision_best, recall_best, f1_best, auc_best))

    # Train with all data by best parameters
    model_dir = os.path.join('Train', 'dnn_model')
    if os.path.exists(model_dir):  # Clean final model dir
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    model = DNNModel(input_size, hidden_units_best).to(device)
    model.apply(weights_init)
    criterion = BCELossWithL1(model, lambda_l1=5e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07, amsgrad=False, weight_decay=1e-5)

    # Ensure the directory exists
    log_dir = os.path.join(model_dir, 'logs')
    if os.path.exists(log_dir):  # Clean final model dir
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    # Proceed with creating the SummaryWriter
    writer = create_writer_safe(log_dir)
    early_stop_counter = 0
    best_auc = 0
    epoch_final = 0
    try:
        for epoch in range(epochs):
            train_loss = 0.0
            eval_loss = 0.0
            eval_accuracy = 0.0
            eval_precision = 0.0
            eval_recall = 0.0
            eval_f1 = 0.0
            # Train the model with all the data
            model.train()
            for batch in dataloader_train:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            # Calculate average train loss
            train_loss /= len(dataloader_train)
            # Evaluate the model's effectiveness
            model.eval()
            with torch.no_grad():
                all_labels = []
                all_outputs = []
                for batch in dataloader_test:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    test_outputs = model(X_batch)
                    test_loss = criterion(test_outputs.squeeze(), y_batch)
                    eval_loss += test_loss.item()
                    # Store outputs and labels for global AUC calculation
                    all_labels.extend(y_batch.cpu().numpy())
                    all_outputs.extend(test_outputs.squeeze().cpu().numpy())

                eval_loss /= len(dataloader_test)

                y_true = np.array(all_labels)
                y_scores = np.array(all_outputs)

                optimal_threshold = find_optimal_threshold(y_true, y_scores)

                y_pred = (y_scores > optimal_threshold).astype(int)

                if len(np.unique(y_true)) > 1:
                    eval_auc = roc_auc_score(y_true, y_scores)
                else:
                    warnings.warn("Only one class present in validation set. ROC AUC score is not defined.")
                    eval_auc = 0.5

                eval_accuracy = accuracy_score(y_true, y_pred)
                eval_precision = precision_score(y_true, y_pred, zero_division=0)
                eval_recall = recall_score(y_true, y_pred, zero_division=0)
                eval_f1 = f1_score(y_true, y_pred, zero_division=0)
                confusion = confusion_matrix(y_true, y_pred)

                metrics = {
                    'Loss/Train': train_loss,
                    'Loss/Evaluate': eval_loss,
                    'Accuracy/Evaluate': eval_accuracy,
                    'Precision/Evaluate': eval_precision,
                    'Recall/Evaluate': eval_recall,
                    'F1/Evaluate': eval_f1,
                    'AUC/Evaluate': eval_auc,
                    'Threshold/Evaluate': optimal_threshold
                }
                safe_write_tensorboard(writer, metrics, epoch + 1)

                logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, "
                             f"Eval Accuracy: {eval_accuracy}, Eval Precision: {eval_precision}, "
                             f"Eval Recall: {eval_recall}, Eval F1: {eval_f1}, Eval AUC: {eval_auc}, "
                             f"Optimal Threshold: {optimal_threshold}")
                # Early stopping
                if (eval_auc > best_auc) and (eval_auc >= 0.5) and (eval_recall > 0.02) and (eval_recall < 1):
                    best_auc = eval_auc
                    early_stop_counter = 0
                    # Save best model state
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_auc': best_auc,
                    }, os.path.join(model_dir, 'best_model.pth'))
                else:
                    early_stop_counter += 1
                    if (early_stop_counter >= patience) and (eval_auc >= 0.5):
                        epoch_final = epoch + 1
                        logging.info(f"Early stopping at epoch {epoch_final}")
                        break
    finally:
        # Close writer explicitly at end of fold
        writer.close()

    # calculate total feature
    pic_heatmap_path = os.path.join(output_train_dir, 'confusion_heatmap_dnn_validation.png')
    # plot heatmap
    plt.figure()
    h = sns.heatmap(
        data=confusion,
        cmap="YlGnBu",
        annot=True,
        fmt='d',
        annot_kws={'size': 12},
        cbar=False,
        xticklabels=['Pass', 'Fail'],
        yticklabels=['Pass', 'Fail']
    )
    cb = h.figure.colorbar(h.collections[0])
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Actual Values', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(pic_heatmap_path, bbox_inches='tight')
    plt.close()

    # Log the results
    if epoch_final == 0:
        epoch_final = epochs
    logging.info("\nAfter training with all the samples, the best results:")
    logging.info(
        "\nHidden Units: {0}\nEpochs: {1}\nAccuracy: {2}\nPercision {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
            str(hidden_units_best), epoch_final, eval_accuracy, eval_precision, eval_recall, eval_f1, eval_auc))
    # Create new row for dataframe
    new_row = {
        'Train Type': 'train all samples',
        'Hidden Units': str(hidden_units_best),
        'Epochs': epoch_final,
        'Accuracy': eval_accuracy,
        'Precision': eval_precision,
        'Recall': eval_recall,
        'F1 Score': eval_f1,
        'AUC': eval_auc
    }
    # Append to dataframe
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([new_row])],
        ignore_index=True
    )
    # Save metrics
    metrics_path = os.path.join('Train', 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    # save the middle parameters
    data_preproc = {
        'id_varname': id_varname,
        'target_varname': target_varname,
        'col_dtypes_dict_use': col_dtypes_dict_use,
        'vars_fea_list_ori': vars_fea_list_ori,
        'vars_fea_cha_list_ori': vars_fea_cha_list_ori,
        'vars_fea_num_list_ori': vars_fea_num_list_ori,
        'vars_fea_cha_list': vars_fea_cha_list,
        'vals_fac_value_pairs': vals_fac_value_pairs,
        'inds_colsleft': inds_colsleft,
        'vars_fea_list': vars_fea_list,
        'num_classes': num_classes,
        'imp': imp,
        'scaler': scaler,
        'hidden_units_best': hidden_units_best,
        'optimal_threshold': optimal_threshold,
        'sam_method': sam_method,
        'batch_size': batch_size,
        'seed': seed
    }
    preproc_path = os.path.join('Train', 'prepro_data.pkl')
    pkl_file = open(preproc_path, 'wb')
    pickle.dump(data_preproc, pkl_file)
    pkl_file.close()
    # Save the model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    datetime_end = datetime.datetime.now()
    datetime_intv = datetime_end - datetime_beg
    logging.info("End train the model! time taken: {}".format(datetime_intv))


def act(data_filename, output_dir):
    ## import 
    import os
    import logging
    import pickle
    import datetime
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn import model_selection, metrics
    from sklearn.metrics import precision_recall_curve

    # set config
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")  # Disable GPU usage
    # the below configs that work for pytorch
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    ## load saved data
    starttime = datetime.datetime.now()
    output_train_dir = os.path.join(output_dir, 'Train')
    output_act_dir = os.path.join(output_dir, 'Act')
    if not os.path.exists(output_act_dir):
        os.makedirs(output_act_dir)

    preproc_path = os.path.join(output_train_dir, 'prepro_data.pkl')
    pkl_file = open(preproc_path, 'rb')
    data_preproc = pickle.load(pkl_file)
    pkl_file.close()

    # logging setting
    logging.root.handlers = []  # Clear existing handlers
    filename_actlog = os.path.join(output_act_dir, 'Act.txt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=filename_actlog,
        filemode='w',
        force=True)
    ## load data and logging
    logging.info("begin loading data.")
    id_varname = data_preproc['id_varname']
    target_varname = data_preproc['target_varname']
    col_dtypes_dict_use = data_preproc['col_dtypes_dict_use']
    vars_list = col_dtypes_dict_use.keys()
    vars_fea_list_ori = data_preproc['vars_fea_list_ori']
    vars_fea_cha_list_ori = data_preproc['vars_fea_cha_list_ori']
    vars_fea_num_list_ori = data_preproc['vars_fea_num_list_ori']
    vars_fea_cha_list = data_preproc['vars_fea_cha_list']
    vals_fac_value_pairs = data_preproc['vals_fac_value_pairs']
    inds_colsleft = data_preproc['inds_colsleft']
    vars_fea_list = data_preproc['vars_fea_list']
    imp = data_preproc['imp']
    scaler = data_preproc['scaler']
    num_classes = data_preproc['num_classes']
    input_size = len(vars_fea_list)
    hidden_units_best = data_preproc['hidden_units_best']
    optimal_threshold = data_preproc['optimal_threshold']
    batch_size = data_preproc['batch_size']
    seed = data_preproc['seed']

    # load data
    ## data preparation
    data_test = pd.read_csv(data_filename, usecols=vars_list, dtype=col_dtypes_dict_use)
    id_test = data_test[id_varname]
    y_test = np.array([val.strip() for val in data_test[target_varname]]).copy().astype(np.int32)
    X_test_num = data_test[vars_fea_num_list_ori].copy()
    X_test_fac = data_test[vars_fea_cha_list_ori].copy()
    # deal with nominail data
    for var in vars_fea_cha_list_ori:
        X_test_fac.loc[:, var] = [val.strip() for val in X_test_fac[var].astype(str)]

    # fill na to other
    X_test_fac = X_test_fac.fillna('other')
    # save values with samples more than 500
    for var in vars_fea_cha_list_ori:
        values = vals_fac_value_pairs[var]
        X_test_fac.loc[~ X_test_fac[var].isin(values), [var]] = 'other'

    # dummy encoding
    for var in vars_fea_cha_list_ori:
        dummies = pd.get_dummies(X_test_fac[var]).rename(columns=lambda x: var + '__' + str(x))
        X_test_fac = pd.concat([X_test_fac, dummies], axis=1)
        X_test_fac.drop([var], inplace=True, axis=1)

    vars_fea_cha_list_left = list(set(vars_fea_cha_list) - set(X_test_fac.columns))
    if len(vars_fea_cha_list_left) > 0:
        for var in vars_fea_cha_list_left:
            X_test_fac.loc[:, var] = 0

    X_test_fac = X_test_fac[vars_fea_cha_list].copy()

    # deal with numerical features
    # imputation
    X_test_num = imp.transform(X_test_num)
    # data standalization
    X_test_num = scaler.transform(X_test_num)
    # merge data
    X_test = np.hstack((X_test_fac.values, X_test_num))
    ## feature selection
    X_test = X_test[:, inds_colsleft]

    class DNNDataset(Dataset):
        def __init__(self, features, labels=None):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            if self.labels is not None:
                return self.features[idx], self.labels[idx]
            return self.features[idx]

    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            features, labels = zip(*batch)
            features = torch.stack(features)
            labels = torch.stack(labels)
            return features, labels
        return torch.stack(batch)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    class DNNModel(nn.Module):
        def __init__(self, input_size, hidden_units):
            super(DNNModel, self).__init__()
            self.dnn_layers = []
            for i, units in enumerate(hidden_units):
                self.dnn_layers.append(nn.Linear(input_size if i == 0 else hidden_units[i - 1], units))
                self.dnn_layers.append(nn.LayerNorm(units))
                self.dnn_layers.append(nn.LeakyReLU(negative_slope=0.05))
                if i == 0:
                    self.dnn_layers.append(nn.Dropout(0.1))
                else:
                    self.dnn_layers.append(nn.Dropout(0.5))
            self.dnn_layers.append(nn.Linear(hidden_units[-1], 1))
            self.dnn_layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*self.dnn_layers)

        def forward(self, x):
            return self.model(x)

    # Create dataset and dataloader
    dataset_test = DNNDataset(X_test)
    generator_test = torch.Generator()
    generator_test.manual_seed(seed)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        generator=generator_test,
        num_workers=0)

    ## predict model
    model_dir = os.path.join(output_train_dir, 'dnn_model')
    model = DNNModel(input_size=input_size, hidden_units=hidden_units_best)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()
    model.to('cpu')  # Ensure the model is on CPU for inference

    # Predict
    predict_result_path = os.path.join(output_act_dir, 'predict.csv')
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader_test:
            X_batch = batch.to('cpu')  # Ensure the data is on CPU
            outputs = model(X_batch).squeeze().numpy()
            all_predictions.extend(outputs)

    # Convert predictions to numpy array
    y_test_1 = np.array(all_predictions)

    # Save predictions
    y_test_0 = 1 - y_test_1
    y_test = (y_test_1 >= optimal_threshold).astype(int)
    dict_prob = dict()
    df_columns = []
    for id_var in id_varname:
        dict_prob[id_var] = id_test[id_var]
        df_columns.append(id_var)
    df_columns.append("predict")
    dict_prob["predict"] = y_test
    dict_prob[str(1)] = y_test_1
    df_columns.append(str(1))
    dict_prob[str(0)] = y_test_0
    df_columns.append(str(0))
    df_proba = pd.DataFrame(dict_prob)
    df_proba.to_csv(predict_result_path, columns=df_columns, index=None, header=True, mode='w')

    logging.info("end exporting data.")
    endtime = datetime.datetime.now()
    secs_all = (endtime - starttime).seconds
    hours = secs_all // 3600
    mins = (secs_all % 3600) // 60
    seconds = (secs_all % 3600) % 60
    runtime = f'total:{secs_all} seconds;  detail:{hours} hours {mins} mins {seconds} seconds'
    logging.info(runtime)


def test(data_filename, output_dir):
    # import
    import os
    import logging
    import pickle
    import datetime
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, precision_recall_curve

    # logging setting
    starttime = datetime.datetime.now()
    output_test_dir = os.path.join(output_dir, 'Test')
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    logging.root.handlers = []  # Clear existing handlers
    filename_testlog = os.path.join(output_test_dir, 'Test.txt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=filename_testlog,
        filemode='w',
        force=True)
    # read id and target
    preproc_path = os.path.join(output_dir, 'Train', 'prepro_data.pkl')
    pkl_file = open(preproc_path, 'rb')
    data_preproc = pickle.load(pkl_file)
    pkl_file.close()
    id_varname = data_preproc['id_varname']
    target_varname = data_preproc['target_varname']
    num_classes = data_preproc['num_classes']
    sam_method = data_preproc['sam_method']
    optimal_threshold = data_preproc['optimal_threshold']
    # Create the dictionaries
    id_dict = {id_var: str for id_var in id_varname}
    target_dict = {target_varname: int}
    # Merge the dictionaries
    merged_dict = id_dict.copy()  # Make a copy of id_dict to avoid modifying the original
    merged_dict.update(target_dict)
    # read predict data
    predict_result_path = os.path.join(output_dir, 'Act', 'predict.csv')
    df_pred = pd.read_csv(predict_result_path, dtype=id_dict)
    # read test data
    real_result_path = os.path.join(data_filename)
    df_real = pd.read_csv(real_result_path, usecols=id_varname + [target_varname], dtype=merged_dict)
    for i in range(num_classes):
        df_real[str(i)] = df_real[target_varname].map(lambda x: 1 if x == i else 0)
    # logging data information
    logging.info('predict sample number: ' + str(df_pred.shape[0]))
    logging.info('real sample number :' + str(df_real.shape[0]))
    # merge data
    df_res = pd.merge(df_pred, df_real, on=id_varname, suffixes=('_PRE', '_REAL'))
    logging.info('merge sample number :' + str(df_res.shape[0]))
    labels_list_real = [str(i) + '_REAL' for i in range(num_classes)]
    labels_list_pre = [str(i) + '_PRE' for i in range(num_classes)]
    y_test = df_res[labels_list_real].to_numpy()
    y_score = df_res[labels_list_pre].to_numpy()

    i = 1
    y_true = y_test[:, int(i)]
    y_scores = y_score[:, int(i)]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    y_predict = (y_scores > optimal_threshold).astype(int)

    test_accuracy = accuracy_score(y_true, y_predict)
    test_precision = precision_score(y_true, y_predict, zero_division=0)
    test_recall = recall_score(y_true, y_predict, zero_division=0)
    test_f1 = f1_score(y_true, y_predict, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    test_auc = auc(fpr, tpr)

    logging.info('Test Results:')
    logging.info(f"Accuracy: {test_accuracy:.4f}")
    logging.info(f"Precision: {test_precision:.4f}")
    logging.info(f"Recall: {test_recall:.4f}")
    logging.info(f"F1 Score: {test_f1:.4f}")
    logging.info(f"AUC: {test_auc:.4f}")
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")

    logging.info(f"Prediction distribution: {np.bincount(y_predict)}")
    logging.info(f"True distribution: {np.bincount(y_true)}")

    pic_roc_path_each = os.path.join(output_test_dir, 'roc_curve_dnn_all.png')
    sns.set_style("darkgrid")
    palette = sns.color_palette("YlGnBu", n_colors=6)
    plt.figure()
    sns.lineplot(x=fpr, y=tpr, color=palette[1], label=f'ROC Curve (area = {test_auc:.3f})', linewidth=3)
    plt.plot([0, 1], [0, 1], color=palette[3], linestyle='--', linewidth=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.savefig(pic_roc_path_each)
    plt.close()

    pic_heatmap_path = os.path.join(output_test_dir, 'confusion_heatmap_dnn_all_test.png')
    confusion = confusion_matrix(y_true, y_predict)

    plt.figure()
    h = sns.heatmap(
        data=confusion,
        cmap="YlGnBu",
        annot=True,
        fmt='d',
        annot_kws={'size': 12},
        cbar=False,
        xticklabels=['Pass', 'Fail'],
        yticklabels=['Pass', 'Fail']
    )
    cb = h.figure.colorbar(h.collections[0])
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Actual Values', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(pic_heatmap_path, bbox_inches='tight')
    plt.close()

    endtime = datetime.datetime.now()
    secs_all = (endtime - starttime).seconds
    hours = secs_all // 3600
    mins = (secs_all % 3600) // 60
    seconds = (secs_all % 3600) % 60
    runtime = f'total:{secs_all} seconds;  detail:{hours} hours {mins} mins {seconds} seconds'
    logging.info(runtime)


if __name__ == "__main__":
    ## static RandomOverSampler
    ## train
    import os
    from config_loader import get_data_dir

    data_dir = get_data_dir()
    dictionary_filename = os.path.join(data_dir, 'sample/static_dict.csv')
    data_filename = os.path.join(data_dir, 'sample/train.csv')
    id_varname = ['code_module', 'code_presentation', 'id_student']
    target_varname = "target"
    output_dir = os.path.join(data_dir, 'output/output_dnn_all')
    sam_method = 'RandomOverSampler'
    cv_folds = 5  # 5
    hidden_units_list = [[10], [20], [40], [80], [120], [160], [10, 5], [20, 10], [40, 20], [80, 40], [120, 60],
                         [160, 80]]
    lr = 1e-4
    batch_size = 32
    epochs = 300  # None
    patience = 2  # 2
    seed = 1
    pearson_cutoff = 1.1  # 0.8
    train(dictionary_filename=dictionary_filename, data_filename=data_filename, id_varname=id_varname,
          target_varname=target_varname, output_dir=output_dir, sam_method=sam_method, cv_folds=cv_folds,
          hidden_units_list=hidden_units_list, lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
          pearson_cutoff=pearson_cutoff, seed=seed)

    ## act
    output_dir = os.path.join(data_dir, 'output/output_dnn_all')
    data_filename = os.path.join(data_dir, 'sample/test.csv')
    act(data_filename, output_dir)

    ## test
    output_dir = os.path.join(data_dir, 'output/output_dnn_all')
    data_filename = os.path.join(data_dir, 'sample/test.csv')
    test(data_filename=data_filename, output_dir=output_dir)
