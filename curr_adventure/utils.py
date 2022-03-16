import jax.numpy as jnp
import jax.random as jrandom

def get_particles(config):
    num_particles = 1
    if config["particle_init"] == "origin":
        particles = [jnp.zeros(config["dim"]) for _ in range(num_particles)]
    else:
        raise ValueError("Does not support given function {}.".format(config["particle_init"]))
    return jnp.array(particles)

def get_config_to_id_map(configs):
    map_dict = {}

    for net_id in configs:
        conf = configs[net_id]
        tmp_dict = map_dict
        for k, v in conf.items():
            if "potential" in k:
                continue
            if isinstance(v, list):
                v = tuple(v)

            if k not in tmp_dict:
                tmp_dict[k] = {}
            if v not in tmp_dict[k]:
                tmp_dict[k][v] = {}
            prev_dict = tmp_dict
            tmp_dict = tmp_dict[k][v]
        prev_dict[k][v] = net_id
    return map_dict

def get_ids(config_to_id_map, config):
    if not isinstance(config_to_id_map, dict):
        return [config_to_id_map]
    p = list(config_to_id_map.keys())[0]

    ids = []
    for c in config_to_id_map[p]:
        if isinstance(config[p], list):
            config_compare = tuple(config[p])
        else:
            config_compare = config[p]
        if (config_compare is None) or (config_compare == c):
            ids += get_ids(config_to_id_map[p][c], config)
    return ids

def different_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    a[:, 14] = 0
    return (a[0] != a[1:]).any(0)

def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict

# @jit
def woordbury_update(A_inv, C_inv, U, V):
    """(A + UCV)^{-1} = A_inv - A_inv U (C_inv + V A_inv U)^{-1} V A_inv"""
    mid_inv = jnp.linalg.inv(C_inv + jnp.dot(V, jnp.dot(A_inv, U)))
    return A_inv - jnp.dot(A_inv, jnp.dot(U, jnp.dot(mid_inv, jnp.dot(V, A_inv))))