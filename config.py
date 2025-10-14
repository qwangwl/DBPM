import configargparse
from utils import str2bool
from datetime import datetime

def get_parser():
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", is_config_file=True, help="config文件路径")
    parser.add_argument("--seed", type=int, default=20, help="随机种子")

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--num_of_class", type=int, default=3, help="定义类别数量, 对于不同数据集类别数不同")

    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss_type', type=str, default='dann')

    parser.add_argument('--dataset_name', type=str, default="seed3", help="数据集")
    parser.add_argument('--session', type=int, default=1, help="定义此次训练的session")
    parser.add_argument("--seed3_path", type=str, default = "E:\\EEG_DataSets\\SEED\\ExtractedFeatures\\")
    parser.add_argument("--seed4_path", type=str, default = "E:\\EEG_DataSets\\SEED_IV\\eeg_feature_smooth\\")
    current_date = datetime.now().strftime("%m%d")
    parser.add_argument("--tmp_saved_path", type=str, default=f"E:\\EEG\\logs\\default\\{current_date}\\")

    parser.add_argument('--saved_model', type=str2bool, default=False, help="当前训练过程是否存储模型")


    parser.add_argument('--pre_epochs', type=int, default=14)
    parser.add_argument('--num_of_s_clusters', type=int, default=15)
    parser.add_argument('--num_of_t_clusters', type=int, default=15)

    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--min_samples", type=int, default=5)

    parser.add_argument("--ablation", type=str, default="PMEEG", choices=["PMEEG", "WithOutPM", "WithOutSSTS", "WithOutPLLoss", "WithOutPretrain", "WithOutTransferLoss", "WithOutMixSource", "TestStartup"], help="Ablation study type")

    parser.add_argument("--pl_weight", type=float, default=0.01)
    parser.add_argument("--noisy_level", type=float, default=0)
    parser.add_argument("--startup", type=int, default=1)

    parser.add_argument("--num_of_source", type=int, default=14, help="Number of source domains to use for training")

    # For DEAP
    parser.add_argument("--deap_path", type=str, default = "E:\\EEG_DataSets\\DEAP")
    parser.add_argument("--emotion", type=str, default="valence")
    parser.add_argument("--feature_name", type=str, default="de")
    parser.add_argument("--window_sec", type=int, default=1)
    parser.add_argument("--step_sec", type=int, default=1)

    return parser