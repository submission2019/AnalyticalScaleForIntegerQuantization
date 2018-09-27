from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely
import torch
from pathlib import Path

class StatisticManager(metaclass=Singleton):
    def __init__(self, folder, load_stats, stats = ['max', 'min', 'std', 'mean', 'kurtosis', 'mean_abs', 'b']):
        self.name = folder
        home = str(Path.home())
        self.folder = os.path.join(home, 'asiq_data/statistics', folder)
        if not load_stats:
            print("Saving statistics to %s" % self.folder)
        self.stats_names = stats
        self.stats = {}
        self.save_stats = not load_stats
        if load_stats:
            stats_file = os.path.join(self.folder, '%s_summary.csv' % self.name)
            assert os.path.exists(stats_file), "Statistics not found, please run with '-sm collect' first"
            self.stats_df = pd.read_csv(stats_file, index_col=0)
        else:
            self.stats_df = None
        pass

    def save_tensor_stats(self, tensor, id):
        stat_arr = []
        # Calculate tensor stats
        for sn in self.stats_names:
            if sn == 'kurtosis':
                t = tensor.view(tensor.shape[0], -1)
                st = torch.mean(((t - t.mean(-1).unsqueeze(-1)) / t.std(-1).unsqueeze(-1))**4, dim=-1) - 3
            elif sn == 'mean_abs':
                t = tensor.view(tensor.shape[0], -1)
                st = torch.mean(t.abs(), dim=-1)
                # st = torch.mean((t - t.mean(-1).unsqueeze(-1)).abs(), dim=-1)
            elif sn == 'b':
                t = tensor.view(tensor.shape[0], -1)
                st = torch.mean(torch.abs(t - t.mean(-1).unsqueeze(-1)), dim=-1)
            else:
                # collect statistics for entire mini batch
                st = getattr(tensor.view(tensor.shape[0], -1), sn)(-1)
                if type(st) is tuple:
                    st = st[0]
            stat_arr.append(st.cpu().numpy())

        # Add to stats dictionary
        if id in self.stats:
            stat_arr = np.vstack(stat_arr).transpose()
            s = np.concatenate([self.stats[id], stat_arr])
            self.stats[id] = s
        else:
            self.stats[id] = np.vstack(stat_arr).transpose()

    def get_tensor_stats(self, id, kind={'min':'mean', 'max':'mean', 'mean': 'mean','std':'mean', 'range':'mean', 'mean_abs':'mean', 'b':'mean'}):
        if self.stats_df is not None:
            # TODO: add different options for min/max
            min_ = self.stats_df.loc[id, '%s_min' % kind['min']]
            max_ = self.stats_df.loc[id, '%s_max' % kind['max']]
            mean_ = self.stats_df.loc[id, '%s_mean' % kind['mean']]
            std_ = self.stats_df.loc[id, '%s_std' % kind['std']]
            range_ = self.stats_df.loc[id, '%s_range' % kind['range']]
            mean_abs_ = self.stats_df.loc[id, '%s_mean_abs' % kind['mean_abs']]
            b_ = self.stats_df.loc[id, '%s_b' % kind['b']]
            return min_, max_, mean_, std_, range_, mean_abs_, b_
        else:
            return None, None, None, None, None, None, None

    def __exit__(self, *args):
        if self.save_stats:
            # Save statistics
            if os.path.exists(self.folder):
                shutil.rmtree(self.folder)
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            all_stats_df = {}
            for s_id in self.stats:
                path = os.path.join(self.folder, '%s.csv' % s_id)
                df = pd.DataFrame(columns=self.stats_names, data=self.stats[s_id])
                df.to_csv(path, index=False)
                all_stats_df[s_id] = df
            self.__save_summry(all_stats_df)

    def __save_summry(self, all_stats_df):
        columns = []
        c_names = ['max', 'min', 'mean', 'std', 'range', 'kurtosis', 'mean_abs', 'b']
        for c in c_names:
            columns.append('min_%s' % c)
            columns.append('mean_%s' % c)
            columns.append('max_%s' % c)

        df_summary = pd.DataFrame(columns=columns)
        for s_id in sorted_nicely(all_stats_df.keys()):
            all_stats_df[s_id]['range'] = all_stats_df[s_id]['max'] - all_stats_df[s_id]['min']
            for c in c_names:
                df_summary.loc[s_id, 'min_%s' % c] = all_stats_df[s_id][c].min()
                df_summary.loc[s_id, 'mean_%s' % c] = all_stats_df[s_id][c].mean()
                df_summary.loc[s_id, 'max_%s' % c] = all_stats_df[s_id][c].max()
        path = os.path.join(self.folder, '%s_summary.csv' % self.name)
        df_summary.to_csv(path, index=True)
