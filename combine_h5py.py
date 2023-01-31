import numpy as np
from databoost.utils.data import find_h5, read_h5, write_h5

def recursively_copy_h5py(d, f):
    for k in f:
        if k == 'imgs':
            continue
        if hasattr(f[k], "keys") and callable(f[k].keys):
            d[k] = recursively_copy_h5py(d[k], f[k]) if k in d else recursively_copy_h5py({}, f[k])
        else:
            d[k] = np.concatenate((d[k], f[k]), axis=0) if k in d else np.array(f[k])
    return d

def recursively_add_dummy(d):
    for k in d:
        if hasattr(d[k], "keys") and callable(d[k].keys):
            d[k] = recursively_add_dummy(d[k])
        else:
            d[k] = np.concatenate((d[k], np.zeros_like(d[k][0])[None]), axis=0)
    return d

if __name__ == '__main__':
    # dataset_dir = ['/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/seed/pick-place-wall' , 
    #                 '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/prior/success']
    
    dataset_dir = [
                    '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/seed/pick-place-wall',

                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/1_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/2_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/3_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/4_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/5_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/6_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/7_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/8_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/9_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/success/10_of_10' ,

                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/1_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/2_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/3_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/4_of_10' ,
                     '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/5_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/6_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/7_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/8_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/9_of_10' ,
                    # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_prior/fail/10_of_10' ,


                    # oracle retrieval 
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/1_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/2_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/3_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/4_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/5_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/6_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/7_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/8_of_10' ,
                   # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/9_of_10' ,
                   # # '/data/jullian-yapeter/DataBoostBenchmark/metaworld/data/grouped_seed/10_of_10' ,
                    

                ]

    all_data = {}
    for cur_dataset_dir in dataset_dir:
        for i, file in enumerate(find_h5(cur_dataset_dir)):
            f = read_h5(file)
            if i%10==0: print(i)

            # append everything to all_data
            all_data = recursively_copy_h5py(all_data, f)
            
            # add reward 1 at completion to only seed dataset
            if '/seed/' in cur_dataset_dir:
                sr = np.copy(f['dones'])
                seed_bool = np.ones_like(f['dones'])
            else:
                sr = np.zeros_like(f['rewards'])
                seed_bool = np.zeros_like(f['dones'])
            all_data['sparse_rewards'] = np.concatenate((all_data['sparse_rewards'], sr), axis=0) if 'sparse_rewards' in all_data else sr
            all_data['seed'] = np.concatenate((all_data['seed'], seed_bool), axis=0) if 'seed' in all_data else seed_bool

            # print('seed' if '/seed/' in cur_dataset_dir else 'prior')
            # print(all_data['dones'][-5:])
            # print(all_data['sparse_rewards'][-5:])
            # print()

    # rename keys for ease of use later
    all_data['dense_rewards'] = np.copy(all_data['rewards'])
    all_data['rewards'] = np.copy(all_data['sparse_rewards'])
    del all_data['sparse_rewards']


    # add a dummy state at the end to sample the final state
    recursively_add_dummy(all_data)

    # mask goal positions
    all_data['observations'][:, -3:] = 0.0

    for k in all_data:
        if k != 'infos':
            print(k, all_data[k][-5:])
    print()

    print(np.where(all_data['dones']))
    print(np.sum(all_data['seed']), len(all_data['dones']))
    write_h5(all_data, '/home/sdass/boosting/data/pick_place_wall/success50_fail50/success50_fail50.h5')
