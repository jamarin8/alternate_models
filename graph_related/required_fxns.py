import subprocess, re, argparse, time, pickle

pipInstall = "pip install pandas networkx graphviz pydotplus pydot neato matplotlib"
process = subprocess.Popen(pipInstall.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import hashlib

df_fraud = pd.read_csv('sql_runner_.csv')
df_clean = pd.read_csv('sql_runner_kjry.csv')

df_fraud_ = df_fraud[df_fraud.email_address.duplicated(keep=False) &
                     ~df_fraud.fuzzy_device_id.duplicated(keep=False)]
df_fraud_ = df_fraud_.iloc[:10, :]

df = pd.concat([df_fraud_, df_clean.sample(frac=20 / 100, random_state=42)]).reset_index()
df.loc[:, 'phone_number'] = df.loc[:, 'phone_number'].ravel().astype(int)
df.loc[:, 'fraud_flag'] = df.loc[:, 'fraud_flag'].apply(lambda x: 1 if x == 'fraud' else 0)

df.loc[(df['index'] == 12) & (df['email_address'] == 'rchad@gmail.com'),
'customer_id'] = 667

df.loc[(df['index'] == 22) & (df['email_address'] == 'vio@optonline.net'),
'customer_id'] = 123

df.loc[(df['index'] == 23) & (df['email_address'] == 'vio@optonline.net'),
'customer_id'] = 999


def create_adjacency_matrix(df):
    data = df
    adj = np.zeros(len(data) * len(data)).reshape(len(data), -1)

    for d in data.iterrows():
        for dd in data.iterrows():
            if (d[0] != dd[0]) and (d[1]['customer_id'] != dd[1]['customer_id']):
                if d[1]['email_address'] == dd[1]['email_address']:
                    adj[d[0], dd[0]] += 1
                if d[1]['phone_number'] == dd[1]['phone_number']:
                    adj[d[0], dd[0]] += 1
                if d[1]['device_id'] == dd[1]['device_id']:
                    adj[d[0], dd[0]] += 1
                if d[1]['fuzzy_device_id'] == dd[1]['fuzzy_device_id']:
                    adj[d[0], dd[0]] += 1
    return adj


def graph_email_network(G, df, edges):
    pos = nx.random_layout(G, seed=43)

    plt.figure(figsize=(11, 7))
    cmap = matplotlib.colors.ListedColormap(['lightgray', 'red'])
    node_colors = df.fraud_flag
    nx.draw_networkx(G, pos=pos, cmap=cmap,
                     node_size=100,
                     node_color=node_colors, width=3, with_labels=True,
                     horizontalalignment='center',
                     verticalalignment='center',
                     edge_color=['blue'],
                     labels=dict(zip(range(len(df)), df['customer_id'].astype(str).str[0:5])))

    text = nx.draw_networkx_edge_labels(G, pos=pos,
                                        edge_labels=edges,
                                        font_color='r')

    for node, t in text.items():
        t.set_rotation(0)
        t.set_clip_on(False)

    plt.show()


def graph_device_id_network(G, df, edges2):
    pos = nx.random_layout(G, seed=43)

    plt.figure(figsize=(11, 7))
    cmap = matplotlib.colors.ListedColormap(['lightgray', 'red'])
    node_colors = df.fraud_flag
    nx.draw_networkx(G, pos=pos, cmap=cmap,
                     node_size=100,
                     node_color=node_colors, width=3, with_labels=True,
                     horizontalalignment='center',
                     verticalalignment='center',
                     edge_color=['blue'],
                     labels=dict(zip(range(len(df)), df['customer_id'].astype(str).str[0:5])))

    text = nx.draw_networkx_edge_labels(G, pos=pos,
                                        edge_labels=edges2,
                                        font_color='r')

    for _, t in text.items():
        t.set_rotation('horizontal')
        t.set_clip_on(False)

    plt.show()


def search_by_email(email):
    return df[df['email_address'] == email]


def search_by_fuzzy_device_id(fuzzy_id):
    return df[df['fuzzy_device_id'].str[0:7] == fuzzy_id]


def search_by_customer_id(cust_id):
    return df[df['customer_id'].astype(str).str[0:5] == cust_id]