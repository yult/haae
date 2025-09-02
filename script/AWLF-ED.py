# encoding:utf-8
def train(dictionary_filename, data_filename, id_varname, target_varname, output_dir, sam_method='RandomOverSampler',
          cv_folds=5, paras_dict={}, lr=1e-4, batch_size=32, epochs=300, patience=2, pearson_cutoff=1.1, seed=1):
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
    from imblearn.over_sampling import RandomOverSampler, SMOTE
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
    vars_fea_tuple = set(dict_use[dict_use['var_type'] == 'tuple']['var_name']) & vars_fea
    vars_fea_cha = (vars_fea - vars_fea_num) - vars_fea_tuple
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
    vars_fea_tuple_list = sorted(list(vars_fea_tuple), key=None, reverse=False)

    # Handle numerical features
    X_train_num = train[vars_fea_num_list].copy()
    X_train_fac = train[vars_fea_cha_list].copy()
    X_train_tuple = train[vars_fea_tuple_list].copy()

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
    X_train_num_fac = X_train_ori[:, inds_colsleft]
    y_train = np.array([val.strip() for val in train[target_varname]]).copy().astype(np.int32)
    static_input_size = X_train_num_fac.shape[1]
    X_train = np.hstack((X_train_num_fac, X_train_tuple))
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
        print("Sorray! There is no such sampling method!")
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
    col_dtypes_dict_use = {var: col_dtypes_dict_use[var] for var in
                           vars_fea_list_ori + vars_fea_tuple_list + id_varname + [target_varname]}
    datetime_end = datetime.datetime.now()
    datetime_intv = datetime_end - datetime_beg
    logging.info("End import and parse the data sets! time taken: {}".format(datetime_intv))
    # define function for model
    logging.info("Begin train the model...")
    datetime_beg = datetime.datetime.now()
    # define dataset and dataloader
    seq_vars = list(vars_fea_tuple)
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=vars_fea_list + seq_vars)
    X_train_num_fac_resampled = X_train_resampled[vars_fea_list].astype(np.float32)
    datetime_end = datetime.datetime.now()
    ## Create feature columns to describe the data
    labels_list = y_train_resampled
    sequence1_list = [torch.tensor([float(x) for x in seq[1:-1].split(',') if x.strip() != ""]) if (
                pd.notna(seq) and seq.strip() != "") else torch.tensor([]) for seq in X_train_resampled[seq_vars[0]]]
    sequence2_list = [torch.tensor([float(x) for x in seq[1:-1].split(',') if x.strip() != ""]) if (
                pd.notna(seq) and seq.strip() != "") else torch.tensor([]) for seq in X_train_resampled[seq_vars[1]]]
    sample_train = [
        (torch.tensor(fea, dtype=torch.float32), seq1, seq2, label)  # Explicit conversion
        for fea, seq1, seq2, label in zip(
            X_train_num_fac_resampled.values,
            sequence1_list,
            sequence2_list,
            labels_list
        )
    ]
    sample_test = sample_train.copy()
    dynamic_max_len_1 = max(len(seq[1]) for seq in sample_train)
    dynamic_max_len_2 = max(len(seq[2]) for seq in sample_train)
    dynamic_max_len = max(dynamic_max_len_1, dynamic_max_len_2)
    dynamic_input_size = 2

    ## define dataset and dataloader
    class StaticDynamicDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list  # (fea, seq1, seq2, label)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            fea, seq1, seq2, label = self.data[idx]
            return fea, seq1, seq2, label

    def collate_fn(batch):
        fea, seq1_list, seq2_list, labels = zip(*batch)

        seq1_list = [torch.nan_to_num(seq, nan=0.0).float() for seq in seq1_list]
        seq2_list = [torch.nan_to_num(seq, nan=0.0).float() for seq in seq2_list]

        seq1_lengths = torch.tensor([len(seq) for seq in seq1_list])
        seq2_lengths = torch.tensor([len(seq) for seq in seq2_list])

        combined_lengths = torch.max(seq1_lengths, seq2_lengths)
        mask = (torch.arange(dynamic_max_len)[None, :] < combined_lengths[:, None]).float()

        padded_seq1 = torch.zeros(len(seq1_list), dynamic_max_len)
        for i, (s, l) in enumerate(zip(seq1_list, seq1_lengths)):
            padded_seq1[i, :l] = s[:l]

        padded_seq2 = torch.zeros(len(seq2_list), dynamic_max_len)
        for i, (s, l) in enumerate(zip(seq2_list, seq2_lengths)):
            padded_seq2[i, :l] = s[:l]

        x_dynamic = torch.stack([padded_seq1, padded_seq2], dim=2)  # [batch, seq_len, 2]
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len]

        x_static = torch.stack(fea)

        return x_static, x_dynamic, mask, torch.tensor(labels)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    ## define Attention-Based EncoderDecoder Model
    class AttentionEncoderDecoderModel(nn.Module):
        def __init__(self, static_input_size, static_hidden_units, dynamic_input_size, dynamic_hidden_size,
                     dynamic_num_layers, fusion_hidden_size):
            super().__init__()
            # Static feature encoder (DNN)
            self.dnn_layers = nn.ModuleList()
            for i, units in enumerate(static_hidden_units):
                self.dnn_layers.append(nn.Linear(static_input_size if i == 0 else static_hidden_units[i - 1], units))
                self.dnn_layers.append(nn.LayerNorm(units))
                self.dnn_layers.append(nn.LeakyReLU(negative_slope=0.05))
                if i == 0:
                    self.dnn_layers.append(nn.Dropout(0.1))
                else:
                    self.dnn_layers.append(nn.Dropout(0.5))
            self.static_encoder = nn.Sequential(*self.dnn_layers)
            # Dynamic feature decoder (LSTM)
            self.lstms = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(dynamic_num_layers):
                input_size = dynamic_input_size if i == 0 else dynamic_hidden_size
                self.lstms.append(nn.LSTM(
                    input_size=input_size,
                    hidden_size=dynamic_hidden_size,
                    batch_first=True
                ))
                self.norms.append(nn.LayerNorm(dynamic_hidden_size))
            self.attention = nn.Sequential(
                nn.Linear(dynamic_hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            # Feature projection
            self.static_proj = nn.Linear(static_hidden_units[-1], fusion_hidden_size)
            self.dynamic_proj = nn.Linear(dynamic_hidden_size, fusion_hidden_size)
            # Feature fusion with weighted combination
            self.weight_layer = nn.Sequential(
                nn.Linear(2 * fusion_hidden_size, 32),
                nn.LeakyReLU(0.05),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Dropout(0.5),
                nn.Linear(fusion_hidden_size, 1),
                nn.Sigmoid()
            )
            # Add sequence masking support
            self.use_length_mask = True

        def forward(self, x_static, x_dynamic, mask):
            # Static feature processing
            static_feat = self.static_encoder(x_static.float())
            static_feat = self.static_proj(static_feat)
            # Dynamic feature processing
            batch_size, seq_len, _ = x_dynamic.size()
            # Process through LSTM layers
            dynamic_out = x_dynamic
            for lstm, ln in zip(self.lstms, self.norms):
                dynamic_out, _ = lstm(dynamic_out)
                dynamic_out = ln(dynamic_out)  # [batch, seq_len, hidden_size]
            # Calculate attention weights
            attn_weights = self.attention(dynamic_out)  # [batch, seq_len, 1]
            # Apply sequence mask to attention weights
            if self.use_length_mask:
                attn_weights = attn_weights * mask.permute(0, 2, 1)  # [batch, seq_len, 1]
            # Normalize attention weights
            attn_weights = torch.softmax(attn_weights, dim=1)
            # Context vector calculation (weighted sum)
            context = torch.bmm(attn_weights.transpose(1, 2), dynamic_out)  # [batch, 1, hidden_size]
            context = context.squeeze(1)  # [batch, hidden_size]
            # Dynamic feature projection
            dynamic_feat = self.dynamic_proj(context)
            # Attention-based fusion
            combined_feat = torch.cat([static_feat, dynamic_feat], dim=1)
            weights = self.weight_layer(combined_feat)
            alpha = weights[:, 0].unsqueeze(1)
            beta = weights[:, 1].unsqueeze(1)
            # Weighted combination
            fused = alpha * static_feat + beta * dynamic_feat
            output = self.fusion(fused).squeeze(-1)
            return torch.clamp(output, 1e-7, 1 - 1e-7), attn_weights  # Prevent numerical instability

    ## define BCELossWithL1
    class BCELossWithL1(nn.Module):
        def __init__(self, model, lambda_l1=5e-3):
            super().__init__()
            self.model = model
            self.lambda_l1 = lambda_l1
            self.bce_loss = nn.BCELoss()

        def forward(self, y_pred, y_true):
            y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
            loss = self.bce_loss(y_pred, y_true)

            l1_reg = torch.tensor(0., device=y_pred.device)
            for param in self.model.parameters():
                l1_reg += torch.sum(torch.abs(param))

            total_loss = loss + self.lambda_l1 * l1_reg
            return total_loss

    def plot_attention_heatmap(weights, masks, output_dir):
        """Plot attention weights heatmap for a single batch"""
        plt.figure(figsize=(12, 6))
        # Apply mask
        valid_weights = (weights * masks.permute(0, 2, 1)).cpu().numpy()
        # Create custom tick labels starting from 1
        seq_len = valid_weights.shape[1]
        xticks = np.arange(1, seq_len + 1)
        ax = sns.heatmap(valid_weights.mean(0),  # Average across batch
                         cmap="YlGnBu",
                         annot=True,
                         fmt=".2f",
                         cbar_kws={'label': 'Attention Intensity'},
                         xticklabels=xticks,
                         yticklabels=False)
        # Customize axes
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Average Attention Weights in the Final Training Batch")
        # Remove y-axis ticks completely
        ax.set_yticks([])
        # Attention-Based Encoder Decoder Attention Weight Distribution
        plt.savefig(os.path.join(output_dir, f"attention_heatmap_awlf_ed.png"), bbox_inches='tight')
        plt.close()

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
    dataset_train = StaticDynamicDataset(sample_train)
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
    dataset_test = StaticDynamicDataset(sample_test)
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
    model_tmp_dir = os.path.join('Train', 'attention-based_encoder_decoder_tmp_model')
    if os.path.exists(model_tmp_dir):  # Clean tmp model directory
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir)
    iter_paras = 0
    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=[
        'Train Type',
        'Fusion Hidden Size',
        'Epochs',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'AUC'
    ])
    static_hidden_units = paras_dict['static_hidden_units']
    dynamic_hidden_size = paras_dict['dynamic_hidden_size']
    dynamic_num_layers = paras_dict['dynamic_num_layers']
    fusion_hidden_list = paras_dict['fusion_hidden_list']
    for fusion_hidden_size in fusion_hidden_list:
        iter_paras += 1
        tmp_dir = os.path.join(model_tmp_dir, str(iter_paras))
        os.makedirs(tmp_dir, exist_ok=True)

        # Define the model, loss function, and optimizer
        model = AttentionEncoderDecoderModel(static_input_size=static_input_size,
                                             static_hidden_units=static_hidden_units,
                                             dynamic_input_size=dynamic_input_size,
                                             dynamic_hidden_size=dynamic_hidden_size,
                                             dynamic_num_layers=dynamic_num_layers,
                                             fusion_hidden_size=fusion_hidden_size).to(device)
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

        for fold, (train_idx, test_idx) in enumerate(kfold.split(sample_train, labels_list)):
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
                dataset_train_folder = StaticDynamicDataset([sample_train[i] for i in train_idx])
                dataset_test_folder = StaticDynamicDataset([sample_test[i] for i in test_idx])
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
                        X_static_batch, X_dynamic_batch, mask, y_batch = batch
                        X_static_batch, X_dynamic_batch, y_batch = X_static_batch.to(device), X_dynamic_batch.to(
                            device), y_batch.float().to(device)
                        optimizer.zero_grad()
                        outputs, attn_weights = model(X_static_batch, X_dynamic_batch, mask)
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
                            X_static_batch, X_dynamic_batch, mask, y_batch = batch
                            X_static_batch, X_dynamic_batch, y_batch = X_static_batch.to(device), X_dynamic_batch.to(
                                device), y_batch.float().to(device)
                            test_outputs, attn_weights = model(X_static_batch, X_dynamic_batch, mask)
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

                    # Use the new safe_write_tensorboard function
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
            'Fusion Hidden Size': fusion_hidden_size,
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
            fusion_hidden_size_best = fusion_hidden_size
            epoch_best = epoch_iter
            accuracy_best = accuracy_iter
            precision_best = precision_iter
            recall_best = recall_iter
            f1_best = f1_iter
            auc_best = auc_iter
            best_model = model
        logging.info(
            "\nFusion Hidden Size: {0}\nEpochs: {1}\nAccuracy: {2}\nPrecision: {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
                fusion_hidden_size, epoch_iter, accuracy_iter, precision_iter, recall_iter, f1_iter, auc_iter))

    # print the best results
    paras_dict['static_input_size'] = static_input_size
    paras_dict['dynamic_input_size'] = dynamic_input_size
    paras_dict['dynamic_max_len'] = dynamic_max_len
    paras_dict_best = {'Fusion Hidden Size': fusion_hidden_size_best, 'Optimal Threshold': optimal_threshold}
    logging.info(
        "\ntrain best parameters:\nFusion Hidden Size: {0}\nEpochs: {1}\nAccuracy: {2}\nPrecision: {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
            fusion_hidden_size_best, epoch_best, accuracy_best, precision_best, recall_best, f1_best, auc_best))
    # Train with all data by best parameters
    model_dir = os.path.join('Train', 'attention-based_encoder_decoder_model')
    if os.path.exists(model_dir):  # Clean final model dir
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    model = AttentionEncoderDecoderModel(static_input_size=static_input_size, static_hidden_units=static_hidden_units,
                                         dynamic_input_size=dynamic_input_size, dynamic_hidden_size=dynamic_hidden_size,
                                         dynamic_num_layers=dynamic_num_layers,
                                         fusion_hidden_size=fusion_hidden_size_best).to(device)
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
    best_mask = None
    best_attn_weights = None
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
                X_static_batch, X_dynamic_batch, mask, y_batch = batch
                X_static_batch = X_static_batch.float().to(device)
                X_dynamic_batch = X_dynamic_batch.float().to(device)
                y_batch = y_batch.float().to(device)
                optimizer.zero_grad()
                outputs, attn_weights = model(X_static_batch, X_dynamic_batch, mask)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            # Calculate average training loss
            train_loss /= len(dataloader_train)
            # Evaluate the model's effectiveness
            model.eval()
            with torch.no_grad():
                all_labels = []
                all_outputs = []
                for batch in dataloader_test:
                    X_static_batch, X_dynamic_batch, mask, y_batch = batch
                    X_static_batch = X_static_batch.float().to(device)
                    X_dynamic_batch = X_dynamic_batch.float().to(device)
                    y_batch = y_batch.float().to(device)
                    test_outputs, attn_weights = model(X_static_batch, X_dynamic_batch, mask)
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
            if (eval_auc > best_auc) and (eval_auc > 0.5) and (eval_recall > 0.02) and (eval_recall < 1):
                best_auc = eval_auc
                best_mask = mask.detach().cpu()
                best_attn_weights = attn_weights.detach().cpu()
                early_stop_counter = 0
                # Save best model state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_auc': best_auc,
                    'best_mask': best_mask,
                    'best_attn_weights': best_attn_weights,
                }, os.path.join(model_dir, 'best_model.pth'))
            else:
                early_stop_counter += 1
                if (early_stop_counter >= patience) and (eval_auc > 0.5):
                    epoch_final = epoch + 1
                    logging.info(f"Early stopping at epoch {epoch_final}")
                    break
    finally:
        # Close writer explicitly at end of fold
        writer.close()
    # Create attention plot
    plot_attention_heatmap(
        best_mask,
        best_attn_weights,
        output_train_dir)
    # calculate total feature
    pic_heatmap_path = os.path.join(output_train_dir, 'confusion_heatmap_awlf_ed_validation.png')
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
        "\nFusion Hidden Size: {0}\nEpochs: {1}\nAccuracy: {2}\nPrecision: {3}\nRecall: {4}\nF1 Score: {5}\nAUC: {6}".format(
            fusion_hidden_size_best, epoch_final, eval_accuracy, eval_precision, eval_recall, eval_f1, eval_auc))
    # Create new row for dataframe
    new_row = {
        'Train Type': 'train all samples',
        'Fusion Hidden Size': fusion_hidden_size_best,
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
        'dynamic_max_len': dynamic_max_len,
        'col_dtypes_dict_use': col_dtypes_dict_use,
        'vars_fea_list_ori': vars_fea_list_ori,
        'vars_fea_cha_list_ori': vars_fea_cha_list_ori,
        'vars_fea_num_list_ori': vars_fea_num_list_ori,
        'vars_fea_cha_list': vars_fea_cha_list,
        'vars_fea_tuple_list': vars_fea_tuple_list,
        'vars_fea_tuple': vars_fea_tuple,
        'vals_fac_value_pairs': vals_fac_value_pairs,
        'inds_colsleft': inds_colsleft,
        'vars_fea_list': vars_fea_list,
        'num_classes': num_classes,
        'imp': imp,
        'scaler': scaler,
        'paras_dict_set': paras_dict,
        'paras_dict_best': paras_dict_best,
        'sam_method': sam_method,
        'batch_size': batch_size,
        'seed': seed
    }
    preproc_path = os.path.join('Train', 'prepro_data.pkl')
    pkl_file = open(preproc_path, 'wb')
    pickle.dump(data_preproc, pkl_file)
    pkl_file.close()
    # Save the model
    safe_save_model(model, os.path.join(model_dir, 'model.pth'))

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
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    vars_fea_num_list_ori = data_preproc['vars_fea_num_list_ori']
    vars_fea_cha_list_ori = data_preproc['vars_fea_cha_list_ori']
    vars_fea_cha_list = data_preproc['vars_fea_cha_list']
    vars_fea_tuple_list = data_preproc['vars_fea_tuple_list']
    vars_fea_tuple = data_preproc['vars_fea_tuple']
    vals_fac_value_pairs = data_preproc['vals_fac_value_pairs']
    inds_colsleft = data_preproc['inds_colsleft']
    vars_fea_list = data_preproc['vars_fea_list']
    imp = data_preproc['imp']
    scaler = data_preproc['scaler']
    num_classes = data_preproc['num_classes']
    batch_size = data_preproc['batch_size']
    seed = data_preproc['seed']
    paras_dict = data_preproc['paras_dict_set']
    static_hidden_units = paras_dict['static_hidden_units']
    static_input_size = paras_dict['static_input_size']
    dynamic_input_size = paras_dict['dynamic_input_size']
    dynamic_hidden_size = paras_dict['dynamic_hidden_size']
    dynamic_num_layers = paras_dict['dynamic_num_layers']
    dynamic_max_len = paras_dict['dynamic_max_len']
    paras_dict_best = data_preproc['paras_dict_best']
    fusion_hidden_size = paras_dict_best['Fusion Hidden Size']
    optimal_threshold = paras_dict_best['Optimal Threshold']

    # load data
    ## data preparation
    data_test = pd.read_csv(data_filename, usecols=vars_list, dtype=col_dtypes_dict_use)
    id_test = data_test[id_varname]
    y_test = np.array([val.strip() for val in data_test[target_varname]]).copy().astype(np.int32)
    X_test_num = data_test[vars_fea_num_list_ori].copy()
    X_test_fac = data_test[vars_fea_cha_list_ori].copy()
    X_test_tuple = data_test[vars_fea_tuple_list].copy()

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
    X_test_num_fac = np.hstack((X_test_fac.values, X_test_num))
    ## feature selection
    X_test_num_fac = X_test_num_fac[:, inds_colsleft]
    X_test = np.hstack((X_test_num_fac, X_test_tuple))
    # build DataLoader
    seq_vars = list(vars_fea_tuple)
    labels_list = y_test
    X_test = pd.DataFrame(X_test, columns=vars_fea_list + seq_vars)
    X_test_num_fac = X_test[vars_fea_list].astype(np.float32)
    sequence1_list = [torch.tensor([float(x) for x in seq[1:-1].split(',') if x.strip() != ""]) if (
                pd.notna(seq) and seq.strip() != "") else torch.tensor([]) for seq in X_test[seq_vars[0]]]
    sequence2_list = [torch.tensor([float(x) for x in seq[1:-1].split(',') if x.strip() != ""]) if (
                pd.notna(seq) and seq.strip() != "") else torch.tensor([]) for seq in X_test[seq_vars[1]]]
    sample_test = [
        (torch.tensor(fea, dtype=torch.float32), seq1, seq2, label)  # Explicit conversion
        for fea, seq1, seq2, label in zip(
            X_test_num_fac.values,
            sequence1_list,
            sequence2_list,
            labels_list
        )
    ]

    ## define dataset and dataloader
    class StaticDynamicDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list  # (fea, seq1, seq2, label)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            fea, seq1, seq2, label = self.data[idx]
            return fea, seq1, seq2, label

    def collate_fn(batch):
        fea, seq1_list, seq2_list, labels = zip(*batch)

        seq1_list = [torch.nan_to_num(seq, nan=0.0).float() for seq in seq1_list]
        seq2_list = [torch.nan_to_num(seq, nan=0.0).float() for seq in seq2_list]

        seq1_lengths = torch.tensor([len(seq) for seq in seq1_list])
        seq2_lengths = torch.tensor([len(seq) for seq in seq2_list])

        combined_lengths = torch.max(seq1_lengths, seq2_lengths)
        mask = (torch.arange(dynamic_max_len)[None, :] < combined_lengths[:, None]).float()

        padded_seq1 = torch.zeros(len(seq1_list), dynamic_max_len)
        for i, (s, l) in enumerate(zip(seq1_list, seq1_lengths)):
            padded_seq1[i, :l] = s[:l]

        padded_seq2 = torch.zeros(len(seq2_list), dynamic_max_len)
        for i, (s, l) in enumerate(zip(seq2_list, seq2_lengths)):
            padded_seq2[i, :l] = s[:l]

        x_dynamic = torch.stack([padded_seq1, padded_seq2], dim=2)  # [batch, seq_len, 2]
        mask = mask.unsqueeze(1)  # [batch, 1, seq_len]

        x_static = torch.stack(fea)

        return x_static, x_dynamic, mask, torch.tensor(labels)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    ## define Attention-Based EncoderDecoder Model
    class AttentionEncoderDecoderModel(nn.Module):
        def __init__(self, static_input_size, static_hidden_units, dynamic_input_size, dynamic_hidden_size,
                     dynamic_num_layers, fusion_hidden_size):
            super().__init__()
            # Static feature encoder (DNN)
            self.dnn_layers = nn.ModuleList()
            for i, units in enumerate(static_hidden_units):
                self.dnn_layers.append(nn.Linear(static_input_size if i == 0 else static_hidden_units[i - 1], units))
                self.dnn_layers.append(nn.LayerNorm(units))
                self.dnn_layers.append(nn.LeakyReLU(negative_slope=0.05))
                if i == 0:
                    self.dnn_layers.append(nn.Dropout(0.1))
                else:
                    self.dnn_layers.append(nn.Dropout(0.5))
            self.static_encoder = nn.Sequential(*self.dnn_layers)
            # Dynamic feature decoder (LSTM)
            self.lstms = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(dynamic_num_layers):
                input_size = dynamic_input_size if i == 0 else dynamic_hidden_size
                self.lstms.append(nn.LSTM(
                    input_size=input_size,
                    hidden_size=dynamic_hidden_size,
                    batch_first=True
                ))
                self.norms.append(nn.LayerNorm(dynamic_hidden_size))
            self.attention = nn.Sequential(
                nn.Linear(dynamic_hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            # Feature projection
            self.static_proj = nn.Linear(static_hidden_units[-1], fusion_hidden_size)
            self.dynamic_proj = nn.Linear(dynamic_hidden_size, fusion_hidden_size)
            # Feature fusion with weighted combination
            self.weight_layer = nn.Sequential(
                nn.Linear(2 * fusion_hidden_size, 32),
                nn.LeakyReLU(0.05),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_hidden_size),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Dropout(0.5),
                nn.Linear(fusion_hidden_size, 1),
                nn.Sigmoid()
            )
            # Add sequence masking support
            self.use_length_mask = True

        def forward(self, x_static, x_dynamic, mask):
            # Static feature processing
            static_feat = self.static_encoder(x_static.float())
            static_feat = self.static_proj(static_feat)
            # Dynamic feature processing
            batch_size, seq_len, _ = x_dynamic.size()
            # Process through LSTM layers
            dynamic_out = x_dynamic
            for lstm, ln in zip(self.lstms, self.norms):
                dynamic_out, _ = lstm(dynamic_out)
                dynamic_out = ln(dynamic_out)  # [batch, seq_len, hidden_size]
            # Calculate attention weights
            attn_weights = self.attention(dynamic_out)  # [batch, seq_len, 1]
            # Apply sequence mask to attention weights
            if self.use_length_mask:
                attn_weights = attn_weights * mask.permute(0, 2, 1)  # [batch, seq_len, 1]
            # Normalize attention weights
            attn_weights = torch.softmax(attn_weights, dim=1)
            # Context vector calculation (weighted sum)
            context = torch.bmm(attn_weights.transpose(1, 2), dynamic_out)  # [batch, 1, hidden_size]
            context = context.squeeze(1)  # [batch, hidden_size]
            # Dynamic feature projection
            dynamic_feat = self.dynamic_proj(context)
            # Attention-based fusion
            combined_feat = torch.cat([static_feat, dynamic_feat], dim=1)
            weights = self.weight_layer(combined_feat)
            alpha = weights[:, 0].unsqueeze(1)
            beta = weights[:, 1].unsqueeze(1)
            # Weighted combination
            fused = alpha * static_feat + beta * dynamic_feat
            output = self.fusion(fused).squeeze(-1)
            return torch.clamp(output, 1e-7, 1 - 1e-7), alpha, beta, attn_weights  # Prevent numerical instability

    def plot_fusion_distribution(all_fuse_alphas, all_fuse_betas, output_dir):
        """Combine all valid attention weights and sequence length distributions for plotting"""

        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = sns.color_palette("YlGnBu", n_colors=2)

        median_alpha = np.median(all_fuse_alphas)
        median_beta = np.median(all_fuse_betas)

        sns.histplot(all_fuse_alphas, bins=50, kde=True, ax=ax, color=palette[0], label='Static Features Fusion Weight',
                     alpha=0.5)
        sns.histplot(all_fuse_betas, bins=50, kde=True, ax=ax, color=palette[1], label='Dynamic Features Fusion Weight',
                     alpha=0.5)

        ax.axvline(median_alpha, color=palette[0], linestyle='--', linewidth=2,
                   label=f'Static Features Fusion Weight Median = {median_alpha:.3f}')
        ax.axvline(median_beta, color=palette[1], linestyle='--', linewidth=2,
                   label=f'Dynamic Features Fusion Weight Median = {median_beta:.3f}')

        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel("Fusion Weight", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Distribution of Fusion Weights", fontsize=14, pad=20)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fusion_weight_awlf_ed.png"),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_temporal_fusion(all_fuse_alphas, all_fuse_betas, all_valid_lengths, output_dir):
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 18))
        palette = sns.color_palette("YlGnBu", n_colors=2)
        plt.rcParams.update({'font.size': 12})
        title_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for i, ax in enumerate(axes):
            target_length = i + 1
            mask = (np.array(all_valid_lengths) == target_length)
            filtered_alphas = [alp for alp, m in zip(all_fuse_alphas, mask) if m]
            filtered_betas = [bet for bet, n in zip(all_fuse_betas, mask) if n]

            if not filtered_alphas or not filtered_betas:
                ax.set_visible(False)
                continue

            sns.histplot(filtered_alphas, bins=50, kde=True, ax=ax, color=palette[0],
                         label='Static Features Fusion Weight')
            sns.histplot(filtered_betas, bins=50, kde=True, ax=ax, color=palette[1],
                         label='Dynamic Features Fusion Weight')

            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xlabel("Fusion Weight", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(f"{title_labels[i]} Distribution of Fusion Weights for Effective Length = {target_length}",
                         fontsize=14, pad=20)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fusion_weight_subplots_awlf_ed.png"), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_attention_distribution(all_attentions, all_valid_lengths, output_dir):
        """Combine all valid attention weights and sequence length distributions for plotting"""
        palette = sns.color_palette("YlGnBu", n_colors=2)
        # Create figure with side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        # Common style settings
        plt.rcParams.update({'font.size': 12})
        # Plot 1: Attention weights distribution
        flat_attentions = np.concatenate([np.array(att) for att in all_attentions])
        sns.histplot(flat_attentions, bins=50, kde=True, ax=ax1, color=palette[0])
        # Add statistics annotations at top
        # Add value labels on bars 
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2.,
                     height + 5,
                     f'{height:.0f}',
                     ha='center',
                     va='bottom',
                     fontsize=10)
        # Plot 2: Sequence length distribution
        sns.histplot(all_valid_lengths,
                     bins=range(1, max(all_valid_lengths) + 2),
                     kde=True, ax=ax2, color=palette[1])
        # Add value labels on bars 
        for p in ax2.patches:
            height = p.get_height()
            ax2.text(p.get_x() + p.get_width() / 2.,
                     height + 3,
                     f'{height:.0f}',
                     ha='center',
                     va='bottom',
                     fontsize=10)
        # Common axis styling
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)
        # Set subplot titles below with 14pt font
        ax1.set_title("(a) Attention Weights Distribution (All Valid Positions)",
                      fontsize=14, y=-0.35, pad=35)
        ax2.set_title("(b) Sequence Length Distribution",
                      fontsize=14, y=-0.35, pad=35)
        # Axis labels
        ax1.set_xlabel("Attention Weight Value", fontsize=12, labelpad=10)
        ax1.set_ylabel("Frequency Count", fontsize=12, labelpad=10)
        ax2.set_xlabel("Sequence Position Count", fontsize=12, labelpad=10)
        ax2.set_ylabel("Sample Frequency", fontsize=12, labelpad=10)
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, wspace=0.3)
        plt.savefig(os.path.join(output_dir, 'attention_distribution_comparison_awlf_ed.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_temporal_attention(all_attentions, all_valid_lengths, max_len, output_dir):
        """Visualize attention patterns by effective sequence length with 6 subplots using boxplots"""
        # Create figure with 6 vertical subplots
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 18))
        palette = sns.color_palette("YlGnBu", n_colors=8)

        # Plot each effective length category
        for i, ax in enumerate(axes):
            target_length = i + 1  # Effective lengths from 1 to 7

            # Filter samples with this effective length
            mask = (np.array(all_valid_lengths) == target_length)
            filtered_attentions = [att for att, m in zip(all_attentions, mask) if m]

            if len(filtered_attentions) == 0:
                ax.axis('off')
                continue

            # Create padded array for boxplot
            padded_att = np.zeros((len(filtered_attentions), max_len))
            for j, att in enumerate(filtered_attentions):
                padded_att[j, :len(att)] = att

            # Prepare data for boxplot
            positions = np.arange(1, target_length + 1)
            boxplot_data = [padded_att[:, pos - 1] for pos in positions]

            # Create boxplot
            box = ax.boxplot(boxplot_data,
                             positions=positions,
                             widths=0.6,
                             patch_artist=True,
                             showfliers=True,
                             showmeans=True,
                             meanline=True)
            # In the boxplot creation loop:
            for pos, line in zip(positions, box['medians']):
                x, y = line.get_xydata()[1]  # Get median position
                ax.text(
                    x,
                    y + 0.03,  # Offset above median line
                    f'{y:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    color='black',  # Use high-contrast color
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # Add background
                )

            # Customize boxplot appearance
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(box[element], color=palette[i + 1])

            # Fill boxes with color
            for patch in box['boxes']:
                patch.set(facecolor=palette[i + 1], alpha=0.3)

            # Configure axes
            ax.set_xlim(0.5, target_length + 0.5)
            ax.set_ylim(0, 1)
            ax.set_xticks(positions)
            ax.set_xlabel("Sequence Position", fontsize=12)
            ax.set_ylabel("Attention Weight", fontsize=12)
            ax.set_title(f"Effective Length = {target_length}", fontsize=12, pad=10)
            title_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
            ax.set_title(f"{title_labels[i]} Boxplot of Attention Weight at Effective Length = {target_length}",
                         fontsize=14, y=-0.6, pad=20)

            # Add horizontal grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.xaxis.grid(False)

            # Add legend for boxplot elements
            if i == 0:  # Only add legend to the first subplot
                ax.legend([box["boxes"][0], box["medians"][0], box["means"][0]],
                          ['Distribution', 'Median', 'Mean'],
                          loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temporal_attention_subplots_awlf_ed.png"),
                    bbox_inches='tight', dpi=300)

    # Create dataset and dataloader
    dataset_test = StaticDynamicDataset(sample_test)
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
    model_dir = os.path.join(output_train_dir, 'attention-based_encoder_decoder_model')
    model = AttentionEncoderDecoderModel(static_input_size=static_input_size, static_hidden_units=static_hidden_units,
                                         dynamic_input_size=dynamic_input_size, dynamic_hidden_size=dynamic_hidden_size,
                                         dynamic_num_layers=dynamic_num_layers,
                                         fusion_hidden_size=fusion_hidden_size).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()
    model.to('cpu')  # Ensure the model is on CPU for inference

    # Predict
    predict_result_path = os.path.join(output_act_dir, 'predict.csv')
    fuse_n_attn_path = os.path.join(output_act_dir, 'fusion_and_attention_weights.csv')
    all_predictions = []
    all_attentions = []
    all_valid_lengths = []
    all_fuse_alphas = []
    all_fuse_betas = []
    all_attention_strings = []
    with torch.no_grad():
        for batch in dataloader_test:
            X_static_batch, X_dynamic_batch, mask, _ = batch  # Ignore labels since we're predicting
            X_static_batch = X_static_batch.float().to(device)
            X_dynamic_batch = X_dynamic_batch.float().to(device)
            outputs, fuse_alpha, fuse_beta, attn_weights = model(X_static_batch, X_dynamic_batch, mask)
            outputs = outputs.squeeze().numpy()
            fuse_alpha = fuse_alpha.squeeze().numpy()
            fuse_beta = fuse_beta.squeeze().numpy()
            all_predictions.extend(outputs)
            all_fuse_alphas.extend(fuse_alpha)
            all_fuse_betas.extend(fuse_beta)
            batch_attentions = attn_weights.squeeze(-1).cpu().numpy()  # [batch, seq_len]
            for i in range(batch_attentions.shape[0]):
                valid_length = int(mask[i].sum().item())
                if valid_length > 0:
                    valid_weights = batch_attentions[i][:valid_length]
                    valid_weights = valid_weights / valid_weights.sum()
                    batch_attentions[i][:valid_length] = valid_weights
            # Format attention weights per sample
            for i in range(batch_attentions.shape[0]):
                valid_length = int(mask[i].sum().item())
                # For visualization storage
                all_attentions.append(batch_attentions[i][:valid_length])
                all_valid_lengths.append(valid_length)
                # For CSV storage
                attn_str = ','.join([f"{x:.4f}" for x in batch_attentions[i][:valid_length]])
                all_attention_strings.append(attn_str)
    dynamic_max_len = data_preproc['dynamic_max_len']
    plot_fusion_distribution(all_fuse_alphas, all_fuse_betas, output_act_dir)
    plot_temporal_fusion(all_fuse_alphas, all_fuse_betas, all_valid_lengths, output_act_dir)
    plot_attention_distribution(all_attentions, all_valid_lengths, output_act_dir)
    plot_temporal_attention(all_attentions, all_valid_lengths, dynamic_max_len, output_act_dir)
    flat_attentions = np.concatenate([np.array(att) for att in all_attentions])
    att_max = flat_attentions.max()
    att_min = flat_attentions.min()
    att_std = flat_attentions.std()

    logging.info(f"Predict fusion weights statistics - "
                 f"Max: {att_max:.4f}, Min: {att_min:.4f}, Std: {att_std:.4f}, "
                 f"Samples: {len(flat_attentions)}")

    # Calculate fusion weights statistics
    alpha_max = np.max(all_fuse_alphas)
    alpha_min = np.min(all_fuse_alphas)
    alpha_med = np.median(all_fuse_alphas)
    alpha_std = np.std(all_fuse_alphas)

    logging.info(f"Predict fusion weights statistics - alpha - "
                 f"Max: {alpha_max:.6f}, Min: {alpha_min:.6f}, Median: {alpha_med:.6f}, Std: {alpha_std:.6f}")

    beta_max = np.max(all_fuse_betas)
    beta_min = np.min(all_fuse_betas)
    bata_med = np.median(all_fuse_betas)
    beta_std = np.std(all_fuse_betas)

    logging.info(f"Predict fusion weights statistics - beta - "
                 f"Max: {beta_max:.6f}, Min: {beta_min:.6f}, Median: {bata_med:.6f}, Std: {beta_std:.6f}")

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
    # Save fusion and attentions weight
    dict_fuse_n_attn = dict()
    df_attn_columns = []
    for id_var in id_varname:
        dict_fuse_n_attn[id_var] = id_test[id_var]
        df_attn_columns.append(id_var)
    df_attn_columns.append("alpha")
    df_attn_columns.append("beta")
    df_attn_columns.append("attention_weights")
    dict_fuse_n_attn["alpha"] = all_fuse_alphas
    dict_fuse_n_attn["beta"] = all_fuse_betas
    dict_fuse_n_attn["attention_weights"] = all_attention_strings
    dict_fuse_n_attn = pd.DataFrame(dict_fuse_n_attn)
    dict_fuse_n_attn.to_csv(fuse_n_attn_path, columns=df_attn_columns, index=None, header=True, mode='w')
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
    paras_dict_best = data_preproc['paras_dict_best']
    optimal_threshold = paras_dict_best['Optimal Threshold']
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

    pic_roc_path_each = os.path.join(output_test_dir, 'roc_curve_awlf_ed.png')
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

    pic_heatmap_path = os.path.join(output_test_dir, 'confusion_heatmap_awlf_ed.png')
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
    ## RandomOverSampler attention-based_encoder_decoder
    ## train
    import os
    from config_loader import get_data_dir

    data_dir = get_data_dir()
    dictionary_filename = os.path.join(data_dir, 'sample/dict.csv')
    data_filename = os.path.join(data_dir, 'sample/train.csv')
    id_varname = ['code_module', 'code_presentation', 'id_student']
    target_varname = "target"
    output_dir = os.path.join(data_dir, 'output/output_awlf_ed')
    sam_method = 'RandomOverSampler'
    cv_folds = 5  # 5
    paras_dict = {
        'static_hidden_units': [120, 60],
        'dynamic_hidden_size': 160,
        'dynamic_num_layers': 2,
        'fusion_hidden_list': [60, 120, 160, 240, 320]
    }
    lr = 1e-4
    batch_size = 32
    epochs = 300  # None
    patience = 2  # 2
    seed = 1
    pearson_cutoff = 1.1  # 0.8
    train(dictionary_filename=dictionary_filename, data_filename=data_filename, id_varname=id_varname,
          target_varname=target_varname, output_dir=output_dir, sam_method=sam_method, cv_folds=cv_folds,
          paras_dict=paras_dict, lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
          pearson_cutoff=pearson_cutoff, seed=seed)

    ## act
    output_dir = os.path.join(data_dir, 'output/output_awlf_ed')
    data_filename = os.path.join(data_dir, 'sample/test.csv')
    act(data_filename, output_dir)

    ## test
    output_dir = os.path.join(data_dir, 'output/output_awlf_ed')
    data_filename = os.path.join(data_dir, 'sample/test.csv')
    test(data_filename=data_filename, output_dir=output_dir)
