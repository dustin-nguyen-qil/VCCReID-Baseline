import logging
import time
import os.path as osp
import torch
from datasets.dataset_loader import build_testloader
from baseline import Inference
from utils.eval_metrics import evaluate, evaluate_with_clothes
from utils.evaluate import extract_vid_feature
from config import CONFIG 
from utils.utils import build_model_name
import numpy as np
import matplotlib.pyplot as plt

model_name = build_model_name()
print(f"Testing Model: {model_name}")

def test(model, queryloader, galleryloader, query, gallery):
    model.cuda()
    model.eval()
    print('======== Extracting query features ========')
    query_features, query_pids, query_camids, query_clothes_ids = extract_vid_feature(
        model=model,
        dataloader=queryloader,
        vid2clip_index=query.vid2clip_index,
        data_length=len(query.dataset),
    )
    print('======== Extracting gallery features ========')
    gallery_features, gallery_pids, gallery_camids, gallery_clothes_ids = extract_vid_feature(
        model,
        galleryloader,
        vid2clip_index=gallery.vid2clip_index,
        data_length=len(gallery.dataset),
    )

    torch.cuda.empty_cache()
    
    m, n = query_features.size(0), gallery_features.size(0)
    distance_matrix = torch.zeros((m, n))
    query_features, gallery_features = query_features.cuda(), gallery_features.cuda()
    # Cosine similarity
    for i in range(m):
        distance_matrix[i] = (-torch.mm(query_features[i:i + 1], gallery_features.t())).cpu()
    distance_matrix = distance_matrix.numpy()
    query_pids, query_camids, query_clothes_ids = query_pids.numpy(),\
          query_camids.numpy(), query_clothes_ids.numpy()
    gallery_pids, gallery_camids, gallery_clothes_ids = gallery_pids.numpy(),\
          gallery_camids.numpy(), gallery_clothes_ids.numpy()

    print("Computing CMC and mAP for Standard setting")
    standard_cmc, standard_mAP = evaluate(distance_matrix, query_pids, gallery_pids,
                        query_camids, gallery_camids)
    
    print("Computing CMC and mAP for same clothes setting")
    sc_cmc, sc_mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='SC')
    
    print("Computing CMC and mAP for cloth-changing setting")
    cc_cmc, cc_mAP = evaluate_with_clothes(distance_matrix,
                                     query_pids,
                                     gallery_pids,
                                     query_camids,
                                     gallery_camids,
                                     query_clothes_ids,
                                     gallery_clothes_ids,
                                     mode='CC')
    
    return (standard_cmc*100, standard_mAP*100, sc_cmc*100, sc_mAP*100, cc_cmc*100, cc_mAP*100)

"""
    Testing
"""

state_dict_path = osp.join(CONFIG.METADATA.SAVE_PATH, model_name)

model = Inference(CONFIG)
model.load_state_dict(torch.load(state_dict_path), strict=False)
queryloader, galleryloader, query, gallery = build_testloader()

(standard_cmc, standard_mAP, sc_cmc, sc_mAP, cc_cmc, cc_mAP) = \
    test(model, queryloader, galleryloader, query, gallery)


print("==============================")

sc_results = f"Same Clothes | R-1: {sc_cmc[0]:.1f} | R-5: {sc_cmc[4]:.1f} | R-10: {sc_cmc[9]:.1f} | mAP: {sc_mAP:.1f}"
print(sc_results)
standard_results = f"Standard | R-1: {standard_cmc[0]:.1f} | R-5: {standard_cmc[4]:.1f} | R-10: {standard_cmc[9]:.1f} | mAP: {standard_mAP:.1f}"
print(standard_results)
cc_results = f"Cloth-changing | R-1: {cc_cmc[0]:.1f} | R-5: {cc_cmc[4]:.1f} | R-10: {cc_cmc[9]:.1f} | mAP: {cc_mAP:.1f}"
print(cc_results)

# Calculate the rank values for the x-axis
ranks = np.arange(1, len(standard_cmc)+1)
ranks = np.arange(1, 41)

# # Plot the CMC curve 
plt.plot(ranks, sc_cmc[:40], '-o', label=sc_results)
plt.plot(ranks, standard_cmc[:40], '-o', label=standard_results)
plt.plot(ranks, cc_cmc[:40], '-x', label=cc_results)

plt.xlabel('Rank')
plt.ylabel('Identification Rate')
plt.title(model_name)
plt.grid(False)
# Save the plot to an output folder
path = f"work_space/output/{model_name[:-4]}.png"
plt.legend()
plt.savefig(path)