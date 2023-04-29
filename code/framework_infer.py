import numpy as np
import os

from dualgcn_baseline_keras2 import MetadataGenerate, CelllineGraphAdjNorm, FeatureExtract, ModelEvaluate
from get_data import path_function, output_paths
from keras.models import load_model
from layers.graph import GraphConv
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

path_dict = path_function('../data') 
log_dict = output_paths('../log')


def launch(path_dict, log_dict):
    ppi_adj_info, drug_feature, data_idx = MetadataGenerate(**path_dict)
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, **path_dict)
    X_drug_feat, X_drug_adj, X_cellline_feat, Y, cancer_type_list=FeatureExtract(data_idx, drug_feature, **path_dict)
    X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
    X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
    X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
    X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
    model = load_model('../checkpoint/MyBestDualGCNModel.h5', custom_objects={'GraphConv': GraphConv})
    cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman, Y_pred = ModelEvaluate(model=model,
                                  X_val=X,
                                  Y_val=Y,
                                  cancer_type_test_list=cancer_type_list,
                                  data_test_idx_current=data_idx,
                                  batch_size=128,
                                  **log_dict)
    return Y_pred, Y

Y_pred,Y = launch(path_dict, log_dict)
print(Y_pred)
print(Y)
    # # import ipdb; ipdb.set_trace()
    # # fdir = Path(__file__).resolve().parent

    # # Model specific params
    # test_batch = args.test_batch

    # # -----------------------------
    # # Create output dir for inference results
    # # IMPROVE_DATADIR = fdir/"improve_data_dir"
    # # INFER_DIR = IMPROVE_DATADIR/"infer"

    # # Outputdir name structure: train_dataset-test_datast
    # # import ipdb; ipdb.set_trace()
    # infer_outdir = fdir/args.infer_outdir
    # os.makedirs(infer_outdir, exist_ok=True)

    # # -----------------------------
    # # Test dataset
    # root_test_data = fdir/args.test_ml_datadir

    # # -----------------------------
    # # Prepare PyG datasets
    # DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    # test_data = TestbedDataset(root=root_test_data, dataset=DATA_FILE_NAME)

    # # PyTorch dataloaders
    # # Note! Don't shuffle the val_loader
    # test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    # # CUDA device from env var
    # print("CPU/GPU: ", torch.cuda.is_available())
    # if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
    #     # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
    #     # the values in python script are reindexed and start from 0.
    #     print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
    #     cuda_name = "cuda:0"
    # else:
    #     cuda_name = args.cuda_name

    # # Load the best model (as determined based val data)
    # device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    # model = modeling().to(device)
    # model_file_name = Path(args.model_dir)/"model.pt"
    # model.load_state_dict(torch.load(model_file_name))
    # model.eval()

    # # Compute raw predictions for val data
    # # import ipdb; ipdb.set_trace()
    # G_test, P_test = predicting(model, device, test_loader)
    # pred = pd.DataFrame({true_col_name: G_test, pred_col_name: P_test})

    # # Concat raw predictions with the cancer and drug ids, and the true values
    # RSP_FNAME = "rsp.csv"
    # rsp_df = pd.read_csv(root_test_data/RSP_FNAME)
    # pred = pd.concat([rsp_df, pred], axis=1)
    # pred = pred.astype({"AUC": np.float32, "True": np.float32, "Pred": np.float32})
    # assert sum(pred[true_col_name] == pred[args.y_col_name]) == pred.shape[0], \
    #     f"Columns {args.y_col_name} and {true_col_name} are the ground truth, and thus, should be the same."

    # # Save the raw predictions on val data
    # pred_fname = "test_preds.csv"
    # pred.to_csv(infer_outdir/pred_fname, index=False)

    # # Get performance scores for val data
    # # TODO:
    # # Should this be a standard in CANDLE/IMPROVE?
    # # Here performance scores/metrics are computed using functions defined in
    # # this repo. Consider to use function defined by the framework (e.g., CANDLE)
    # # so that all DRP models use same methods to compute scores.
    # ## Method 1 - compute scores using the loaded model and val data
    # mse_test = mse(G_test, P_test)
    # rmse_test = rmse(G_test, P_test)
    # pcc_test = pearson(G_test, P_test)
    # scc_test = spearman(G_test, P_test)
    # test_scores = {"mse": float(mse_test),
    #               "rmse": float(rmse_test),
    #               "pcc": float(pcc_test),
    #               "scc": float(scc_test)}
    # ## Method 2 - get the scores that were ealier computed (in for loop)
    # # val_scores = {"val_loss": float(best_mse),
    # #               "rmse": float(best_rmse),
    # #               "pcc": float(best_pearson),
    # #               "scc": float(best_spearman)}

    # # Performance scores for Supervisor HPO
    # with open(infer_outdir/"test_scores.json", "w", encoding="utf-8") as f:
    #     json.dump(test_scores, f, ensure_ascii=False, indent=4)

    # print("Scores:\n\t{}".format(test_scores))
    # return test_scores
    
# def run(gParameters):
#     print("In Run Function:\n")
#     args = candle.ArgumentStruct(**gParameters)
#     modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]

#     # Call launch() with specific model arch and args with all HPs
#     scores = launch(modeling, args)

#     # Supervisor HPO
#     with open(Path(args.output_dir) / "scores.json", "w", encoding="utf-8") as f:
#         json.dump(scores, f, ensure_ascii=False, indent=4)

#     return scores


# def initialize_parameters():
#     """ Initialize the parameters for the GraphDRP benchmark. """
#     print("Initializing parameters\n")
#     graphdrp_bmk = bmk.BenchmarkGraphDRP(
#         filepath=bmk.file_path,
#         defmodel="graphdrp_default_model.txt",
#         # defmodel="graphdrp_model_candle.txt",
#         framework="pytorch",
#         prog="GraphDRP",
#         desc="CANDLE compliant GraphDRP",
#     )
#     gParameters = candle.finalize_parameters(graphdrp_bmk)
#     return gParameters


# def main():
#     gParameters = initialize_parameters()
#     print(gParameters)
#     scores = run(gParameters)
#     print("Done inference.")


# if __name__ == "__main__":
#     main()