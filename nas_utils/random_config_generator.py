from collections import defaultdict
import copy
import random


def bin_and_merge(data_dict, num_bins):
    keys = list(data_dict.keys())
    
    # Determine the range of keys
    min_val = min(keys)
    max_val = max(keys)
    
    # Calculate bin width based on the range and number of bins
    bin_width = (max_val - min_val) / num_bins
    
    # Create bins
    bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    # Dictionary to store merged lists for each bin
    merged_dict = defaultdict(list)
    
    # Allocate keys to bins and merge the lists
    for key in keys:
        for i in range(1, len(bin_edges)):
            if bin_edges[i-1] <= key < bin_edges[i]:
                bin_range = (bin_edges[i-1], bin_edges[i])
                merged_dict[i].extend(data_dict[key])
                break
    
    return merged_dict




class RandomCandGenerator():
    def __init__(self, low_rank_search_space):
        size_config = []
        containers = []
        for local in low_rank_search_space._search_space:
            container = defaultdict(list)
            for cfg in local.enumerations:
                container[sum(cfg)].append(cfg)
            container = bin_and_merge(container, 9)
            containers.append(container)
            size_config.append(list(container.keys()))
            
        self.size_config              = size_config
        self.num_candidates_per_block = len(size_config[0])
        self.config_length            = len(size_config)
        self.containers = containers
        self.m = defaultdict(list)
        #random.seed(seed)
        v = []
        self.rec(v, self.m)
        
    def calc(self, v):
        res = 0
        for i in range(self.num_candidates_per_block):
            res += i * v[i]
        return res

    def rec(self, v, m, idx=0, cur=0):
        if idx == (self.num_candidates_per_block-1) :
            v.append(self.config_length - cur)
            m[self.calc(v)].append(copy.copy(v))
            v.pop()
            return

        i = self.config_length - cur
        while i >= 0:
            v.append(i)
            self.rec(v, m, idx+1, cur+i)
            v.pop()
            i -= 1
            
    def random(self):
        row = random.choice(random.choice(self.m))
        ratios = []
        for num, ratio in zip(row, [i for i in range(self.num_candidates_per_block)]):
            ratios += [ratio] * num
        random.shuffle(ratios)
        res = []
        for idx, ratio in enumerate(ratios):
            target_size = self.size_config[idx][ratio] if ratio < len(self.size_config[idx]) else self.size_config[idx][-1]
            res += list(random.choice(self.containers[idx][target_size]))
            
            
        return res