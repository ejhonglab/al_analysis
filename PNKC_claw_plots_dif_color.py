import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from neuprint import Client, NeuronCriteria as NC
import navis.interfaces.neuprint as navisnp
import navis
import plotly.offline as py
import plotly.graph_objects as go

# Conversion constant
PIXEL_TO_UM = 8 / 1000  # 8 nm per pixel = 0.008 µm

# Load neuPrint token
with open("/home/henryhalhtp/auth/neuprint_token.json") as f:
    token = json.load(f)["token"]

client = Client('https://neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token)

# Load pre-clustered synapse data and pre-claw centroids
df_pre_syn = pd.read_csv("pre_clustered_synapses_r=1_65.csv")
df_pre_centroids = pd.read_csv("pre_claw_centroids_r=1_65.csv")

# Loop through KCs at steps of 10
for i in range(1, 101, 10):
    kc_id = df_pre_centroids.iloc[i]['bodyId_post']
    print("Selected KC ID:", kc_id)

    # Get claw centroids for this KC (convert to pixels)
    kc_claws = df_pre_centroids[df_pre_centroids['bodyId_post'] == kc_id].copy()
    kc_claws[['x_pre', 'y_pre', 'z_pre']] = kc_claws[['x_pre', 'y_pre', 'z_pre']] / PIXEL_TO_UM

    # Get PN IDs and synapses for this KC
    df_kc_pre = df_pre_syn[df_pre_syn['bodyId_post'] == kc_id]
    pn_ids = df_kc_pre['bodyId_pre'].dropna().astype(int).unique().tolist()

    # Fetch skeletons
    pn_criteria = NC(bodyId=pn_ids)
    pn_skel = navisnp.fetch_skeletons(pn_criteria, client=client, with_synapses=False)
    kc_skel = navisnp.fetch_skeletons(int(kc_id), client=client, with_synapses=False)

    # Assign distinct colors to PNs using colormap
    cmap = plt.cm.cool
    colors = [mcolors.rgb2hex(cmap(i / len(pn_skel))) for i in range(len(pn_skel))]

    # Apply colors to PN skeletons
    for idx, pn in enumerate(pn_skel):
        pn.color = colors[idx]

    # Set KC color
    kc_skel[0].color = 'black'

    # Combine skeletons
    all_neurons = list(pn_skel) + list(kc_skel)
    all_colors = colors + ['black']
    fig3d = navis.plot3d(all_neurons, backend='plotly', inline=False, colors=all_colors)

    # Add claw centroids (orange spheres)
    claw_spheres = go.Scatter3d(
        x=kc_claws['x_pre'],
        y=kc_claws['y_pre'],
        z=kc_claws['z_pre'],
        mode='markers',
        marker=dict(
            size=6,
            color='orange',
            opacity=0.85
        ),
        name='KC Pre-Claws'
    )
    fig3d.add_trace(claw_spheres)

    # Add PN synapses (colored by PN ID)
    pn_color_map = dict(zip(pn_ids, colors))
    for pn_id in pn_ids:
        syn_df = df_kc_pre[df_kc_pre['bodyId_pre'] == pn_id][['x_pre', 'y_pre', 'z_pre']] / PIXEL_TO_UM
        syn_trace = go.Scatter3d(
            x=syn_df['x_pre'],
            y=syn_df['y_pre'],
            z=syn_df['z_pre'],
            mode='markers',
            marker=dict(
                size=3,
                color=pn_color_map.get(pn_id, 'gray'),
                opacity=0.9
            ),
            name=f'Synapses from PN {pn_id}'
        )
        fig3d.add_trace(syn_trace)

    # Save figure
    output_file = f"PN_KC_{int(kc_id)}_xyz_3D_coloredsyns.html"
    py.plot(fig3d, filename=output_file, auto_open=False)
    print(f"Saved plot: {output_file}")
