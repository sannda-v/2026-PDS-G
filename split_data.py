import pandas as pd
from sklearn.model_selection import StratifiedKFold
import glob
import numpy as np
from pathlib import Path
import string

PATH_RAW = Path("./data")
PATH_PROCESSED = Path("./data/group_data")
NB_GROUP = 15

PATH_PROCESSED.mkdir(parents=True, exist_ok=True)

df_labels = pd.read_csv(f"{PATH_RAW}/metadata.csv")

lst_masks_avail = [mask_path.split("/")[-1].replace("_mask","") for mask_path in glob.glob(f"{PATH_RAW}/masks/*.png")]
df_labels = df_labels[[img_id in lst_masks_avail for img_id in df_labels["img_id"]]]

stratified_kfold = StratifiedKFold(n_splits=NB_GROUP,shuffle=True,random_state=1907)

grp_id = 0
lst_grp = np.zeros_like(df_labels["diagnostic"])
for _,grp_idx in stratified_kfold.split(df_labels,df_labels["diagnostic"]):
    lst_grp[grp_idx] = string.ascii_uppercase[grp_id]
    grp_id +=1

df_labels["group_id"] = lst_grp
df_labels.to_csv(f"{PATH_RAW}/metadata_with_group.csv")
