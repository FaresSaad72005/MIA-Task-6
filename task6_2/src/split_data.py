import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_data(df):

    # 15% Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["image"]))

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # remaining 85% (Train 70% and Val 15%)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)  # 17.65% of the 85% is 15% of the original data set
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["image"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)


    # printing results
    print("=== Split Summary ===")
    print(f"Train Set: {len(train_df)} captions across {train_df['image'].nunique()} images")
    print(f"Val Set:   {len(val_df)} captions across {val_df['image'].nunique()} images")
    print(f"Test Set:  {len(test_df)} captions across {test_df['image'].nunique()} images")

    # Ensure no image appears in more than one set
    train_imgs = set(train_df["image"])
    val_imgs = set(val_df["image"])
    test_imgs = set(test_df["image"])


    return train_df, val_df, test_df
