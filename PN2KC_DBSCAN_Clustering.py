import pandas as pd
from sklearn.cluster import DBSCAN

# minimum confidence level
c = 0.3

# Conversion factor: 1 “hemibrain‐pixel” = 8 nm = 0.008 µm
PIXEL_TO_UM = 8 / 1000

# 1) Load the merged synapse table
# TODO TODO what was this generated from? where is it?
merged = pd.read_csv('merged.csv')

merged = merged[(merged['confidence_pre'] >= 0.3) & (merged['confidence_post'] >= 0.3)]

# 2) Convert *all* pre- and post-synapse coords from pixels → µm
for axis in ['x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post']:
    merged[axis] = merged[axis] * PIXEL_TO_UM

# 3) Save the fully converted table
merged.to_csv('merged_converted.csv', index=False)

# ===============================
# 4) Cluster PRE-synapses (“pre‐claws”)
# ===============================
clustered_pre = []
eps_um      = 1.65
min_samples = 3

for post_id, df_kc in merged.groupby('bodyId_post', sort=False):
    # prepare the pre‐synapse point cloud in µm
    X_pre = df_kc[['x_pre','y_pre','z_pre']].to_numpy()

    db = DBSCAN(eps=eps_um, min_samples=min_samples).fit(X_pre)

    df_kc = df_kc.copy()
    df_kc['pre_cluster'] = db.labels_
    clustered_pre.append(df_kc)

pre_clustered_df = pd.concat(clustered_pre, ignore_index=True)

# Summarize “pre‐claw” membership and centroids
pre_claw_members   = (
    pre_clustered_df
      .groupby(['bodyId_post','pre_cluster'])['bodyId_pre']
      .unique()
      .reset_index(name='pre_cell_ids')
)
pre_claw_centroids = (
    pre_clustered_df[pre_clustered_df['pre_cluster'] != -1]
      .groupby(['bodyId_post','pre_cluster'])[['x_pre','y_pre','z_pre']]
      .mean()
      .reset_index()
)

# Save PRE-clustering results
pre_clustered_df.to_csv('pre_clustered_synapses_r=1_65.csv',   index=False)
pre_claw_members.to_csv('pre_claw_members_r=1_65.csv',         index=False)
pre_claw_centroids.to_csv('pre_claw_centroids_r=1_65.csv',    index=False)

# ===============================
# 5) Cluster POST-synapses (“post‐claws”)
# ===============================
clustered_post = []
for post_id, df_kc in merged.groupby('bodyId_post', sort=False):
    # prepare the post‐synapse point cloud in µm
    X_post = df_kc[['x_post','y_post','z_post']].to_numpy()

    db = DBSCAN(eps=eps_um, min_samples=min_samples).fit(X_post)

    df_kc = df_kc.copy()
    df_kc['post_cluster'] = db.labels_
    clustered_post.append(df_kc)

post_clustered_df = pd.concat(clustered_post, ignore_index=True)

# Summarize “post‐claw” membership and centroids
post_claw_members   = (
    post_clustered_df
      .groupby(['bodyId_post','post_cluster'])['bodyId_pre']
      .unique()
      .reset_index(name='pre_cell_ids')
)
post_claw_centroids = (
    post_clustered_df[post_clustered_df['post_cluster'] != -1]
      .groupby(['bodyId_post','post_cluster'])[['x_post','y_post','z_post']]
      .mean()
      .reset_index()
)


# Save POST-clustering results
post_clustered_df.to_csv('post_clustered_synapses_r=1_65.csv',   index=False)
post_claw_members.to_csv('post_claw_members_r=1_65.csv',         index=False)
post_claw_centroids.to_csv('post_claw_centroids_r=1_65.csv',    index=False)

# ===============================
# 6) Print summary statistics
# ===============================
def summary(df, label_col):
    valid = df[df[label_col] != -1]
    claws_per_kc = valid.groupby('bodyId_post')[label_col].nunique()
    print(f"\n=== Summary for {label_col} ===")
    print("Claws per KC:\n", claws_per_kc)
    print("Grand total claws:", claws_per_kc.sum())
    print(f"Average claws per KC: {claws_per_kc.mean():.2f}")

summary(pre_clustered_df, 'pre_cluster')
summary(post_clustered_df, 'post_cluster')
