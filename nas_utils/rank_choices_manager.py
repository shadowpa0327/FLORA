import random
from itertools import product

class PerBlockRankChoicesContainer:
    def __init__(self, choices, is_non_uniform = True, inclusion = None):
        self.choices = choices
        self.enumerations = list(product(*choices))
        
        if inclusion:
            self.enumerations = [enum for enum in self.enumerations if enum in inclusion]
            if (1.0, 1.0, 1.0) not in self.enumerations:
                self.enumerations.append((1.0, 1.0, 1.0))
        self.enumerations = sorted(self.enumerations, key=lambda x: sum(x))

        if not is_non_uniform:
            self.weights = [1 for _ in range(len(self.enumerations))]
        else:
            self.weights = []
            for enum in self.enumerations:
                val = 1
                for r in enum:
                    val *= r
                self.weights.append(1/val)
            #self.weights = [1/sum(enum) for enum in self.enumerations]
        
    def get_random(self):
        return random.choices(self.enumerations, weights=self.weights)[0]
    
    def get_largest(self):
        return self.enumerations[-1]
    
    def get_smallest(self):
        return self.enumerations[0]

    def get_all(self):
        return self.enumerations

    def check_legal(self, cfg):
        return tuple(cfg) in self.enumerations

class LowRankSearchSpace:
    def __init__(self, rank_choices, num_blocks, choices_per_blocks = 3, is_non_uniform = True, per_block_searched_configs = None):
        self._search_space = []
        for i in range(num_blocks):
            
            if per_block_searched_configs is not None:
                per_block_cfg = []
                for cfg in per_block_searched_configs[i]:
                    per_block_cfg.append(cfg[i * choices_per_blocks: (i+1) * choices_per_blocks])
            else:
                per_block_cfg = None
                
            self._search_space.append(PerBlockRankChoicesContainer(
                rank_choices[i*3 : (i+1) * 3],
                is_non_uniform = is_non_uniform,
                inclusion = per_block_cfg
            ))
            
        self.num_blocks = num_blocks
        self.choices_per_blocks = choices_per_blocks
        self.total_choices = self.num_blocks * self.choices_per_blocks
        self.rank_choices = rank_choices
        self.largest_cfg = [choices[-1] for choices in self.rank_choices ]
        self.smallest_cfg = [choices[-1] for choices in self.rank_choices ]
    def random(self):
        cfg = []
        for i in range(self.num_blocks):
            cfg += self._search_space[i].get_random()
        assert len(cfg) == self.total_choices
        return cfg

    def random_ith_block(self, idx):
        cfg = [1.0] * self.total_choices
        random_block_cfg = self._search_space[idx].get_random()
        cfg[idx * self.choices_per_blocks: (idx + 1) * self.choices_per_blocks] = random_block_cfg
        return cfg

    def get_largest_config(self):
        cfg = []
        for i in range(self.num_blocks):
            cfg += self._search_space[i].get_largest()
        return cfg
        
    def get_smallest_config(self):
        cfg = []
        for i in range(self.num_blocks):
            cfg += self._search_space[i].get_smallest()
        return cfg
    
    
    def get_smallest_config_ith_block(self, idx):
        cfg = [1.0] * self.total_choices
        smallest_block_cfg = self._search_space[idx].get_smallest()
        cfg[idx * self.choices_per_blocks: (idx + 1) * self.choices_per_blocks] = smallest_block_cfg
        return cfg
    
    def get_all_config_ith_block(self, idx):
        all_per_block_configs = self._search_space[idx].get_all()
        all_configs = []
        for per_block_cfg in all_per_block_configs:
            cfg = [1.0] * self.total_choices
            cfg[idx * self.choices_per_blocks: (idx + 1) * self.choices_per_blocks] = per_block_cfg
            all_configs.append(cfg)
        return all_configs
    
    def check_legal(self, cfg):
        for i in self.num_blocks:
            per_block_cfg = cfg[i * self.choices_per_blocks : (i+1) * self.choices_per_blocks]
            if not self._search_space[i].check_legal(per_block_cfg):
                return False
        return True
                
        
    
def build_low_rank_search_space(args, config, force_uniform = False):
    per_block_searched_configs = None
    if config.NAS.LSSS.SEARCHED_CFG_PATH:
        import pickle
        with open(config.NAS.LSSS.SEARCHED_CFG_PATH, 'rb') as file:
            per_block_searched_configs = pickle.load(file)
            
    return LowRankSearchSpace(
        rank_choices = config.NAS.SEARCH_SPACE,
        num_blocks = config.NAS.NUM_BLOCKS,
        choices_per_blocks = config.NAS.NUM_CHOICES_PER_BLOCKS,
        is_non_uniform = False if force_uniform else config.NAS.NON_UNIFORM,
        per_block_searched_configs = per_block_searched_configs
    ) 


if __name__ == '__main__':
    # Example usage
    choices = [[0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625], [0.229167, 0.3125, 0.395833, 0.479167, 0.5625]]
    rank_space = LowRankSearchSpace(choices, 12)

    choices = rank_space.get_smallest_config_ith_block(1)
    print(choices)
    print(len(choices))

