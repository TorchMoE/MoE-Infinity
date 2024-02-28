from collections import Counter
import numpy as np
import copy
from typing import Dict, Set, List, Tuple

from moe_infinity.memory.expert_entry import ExpertCacheEntry, ExpertTraceEntry

decay_from_first = lambda x, L: -1 / L * x + 1
decay_from_last = lambda x, L: 1 / (L+1) * x

layer_decay = lambda x, l: (x+1) / np.abs(l-x + 1)

# def topology_score(score: np.ndarray, current_layer, total_layer):
#     topo_score = copy.deepcopy(score)
#     for i in range(0, current_layer):
#         topo_score[i] = topo_score[i] * decay_from_first(current_layer, total_layer)
#     for i in range(current_layer, total_layer):
#         topo_score[i] = topo_score[i] * decay_from_last(current_layer, total_layer)
#     return topo_score

# def topology_score(entry: ExpertCacheEntry, current_layer, total_layer):
#     if entry.layer_idx < current_layer:
#         return entry.r * decay_from_first(current_layer, total_layer)
#     else:
#         return entry.r * decay_from_last(current_layer, total_layer)
    
#     assert False, "Should not reach here"

# def cross_sequence_score(score_list):
#     seq_score = np.zeros_like(score_list[0][1])
#     freq_sum = 0

#     for freq, score in score_list:
#         freq_sum += freq
#     for freq, score in score_list:
#         seq_score += score * freq / freq_sum

#     return seq_score
    
# def cross_iteration_score(score_list):
#     iter_score = np.zeros_like(score_list[0][1])
#     freq_sum = 0

#     for freq, score in score_list:
#         freq_sum += freq
#     for freq, score in score_list:
#         iter_score += score * freq / freq_sum

#     return iter_score

def convert_score_matrix_to_list(score_matrix: np.ndarray):
    score_list = []
    for layer_idx, layer in enumerate(score_matrix):
        for expert_idx, r in enumerate(layer):
            if r > 0:
                score_list.append(ExpertCacheEntry(expert_idx, layer_idx, r))
    return score_list

def lru_score(cache_entries: Set[ExpertCacheEntry]):
    lru_score = []
    for entry in cache_entries:
        lru_score.append(ExpertCacheEntry(entry.expert_idx, entry.layer_idx, entry.timestamp))
    return lru_score

def lru_score_with_layers(cache_entries: Set[ExpertCacheEntry], current_layer):
    lru_score = []
    for entry in cache_entries:
        if entry.layer_idx >= current_layer and entry.layer_idx < current_layer + 3:
            lru_score.append(ExpertCacheEntry(entry.expert_idx, entry.layer_idx, 1e10))
        else:
            lru_score.append(ExpertCacheEntry(entry.expert_idx, entry.layer_idx, entry.timestamp))
    return lru_score

def lfu_score(expert_freq: dict):
    # convert to list of tuples
    sum = 0
    for key, value in expert_freq.items():
        sum += value

    if sum == 0:
        sum = 1
    
    lfu_score = []
    for key, value in expert_freq.items():
        expert_idx, layer_idx = key
        lfu_score.append(ExpertCacheEntry(expert_idx, layer_idx, value / sum))
    
    return lfu_score

def oracle_score(expert_freq: dict, decoder_entry: ExpertTraceEntry):
    frequency_score = np.zeros_like(decoder_entry.matrix)
    frequency_sum = 0

    for key, value in expert_freq.items():
        frequency_score[key[1], key[0]] = value
        frequency_sum += value

    if frequency_sum == 0:
        frequency_sum = 1
        frequency_score = np.ones_like(frequency_score)

    frequency_score = frequency_score / frequency_sum + 1e-6

    return convert_score_matrix_to_list(frequency_score)

def priority_score(expert_freq: dict, cache_entries: Set[ExpertCacheEntry], trace_entries: Set[ExpertTraceEntry], decoder_entry: ExpertTraceEntry, current_layer, total_layer):
    num_encoder_layers = total_layer // 2
    # print("Cache entries size", len(cache_entries))
    frequency_score = np.zeros_like(decoder_entry.matrix)
    frequency_sum = 0
    for key, value in expert_freq.items():
        frequency_score[key[1], key[0]] = value
        frequency_sum += value

    if np.sum(frequency_score[num_encoder_layers:]) == 0:
        frequency_score[num_encoder_layers:] = 1
    
    if np.sum(frequency_score[:num_encoder_layers]) == 0:
        frequency_score[:num_encoder_layers] = 1

    # frequency_score = frequency_score / frequency_sum + 1e-6
    # frequency_score[:num_encoder_layers] = frequency_score[:num_encoder_layers] / np.sum(frequency_score[:num_encoder_layers]) + 1e-6
    # frequency_score[num_encoder_layers:] = frequency_score[num_encoder_layers:] / np.sum(frequency_score[num_encoder_layers:]) + 1e-6
    frequency_score = frequency_score / np.sum(frequency_score) + 1e-6
    assert np.sum(frequency_score) > 0, f"frequency_score = {frequency_score}, frequency_sum = {frequency_sum}"
    # print("frequency_score", np.sum(frequency_score), np.max(frequency_score), np.min(frequency_score), frequency_score.shape)

    topo_expert_score = np.zeros_like(decoder_entry.matrix)
    for i in range(topo_expert_score.shape[0]):
        entry_layer_idx = i
        if current_layer < num_encoder_layers:
            if i < num_encoder_layers:
                topo_expert_score[i, :] = decay_from_first(i, num_encoder_layers) if i > current_layer else 1.0
            else:
                topo_expert_score[i, :] = decay_from_last(i-num_encoder_layers, num_encoder_layers)
        else:
            if i < num_encoder_layers:
                topo_expert_score[i, :] = decay_from_first(i, num_encoder_layers)
            else:
                topo_expert_score[i, :] = decay_from_last(i-num_encoder_layers, num_encoder_layers) if i > current_layer else 1.0
    # topo_expert_score[:num_encoder_layers, :] = topo_expert_score[:num_encoder_layers, :] * 1.2
    # for entry in cache_entries:
    #     # print(entry.layer_idx, current_layer, total_layer)
    #     entry_layer_idx = entry.layer_idx
    #     current_layer_idx = current_layer
    #     if entry_layer_idx > num_encoder_layers:
    #         entry_layer_idx -= num_encoder_layers
    #     if current_layer_idx > num_encoder_layers:
    #         current_layer_idx -= num_encoder_layers
    #     L = num_encoder_layers

    #     assert entry_layer_idx >= 0 and entry_layer_idx <= L, f"entry_layer_idx = {entry_layer_idx}, total_layer = {L}"

    #     # if entry_layer_idx < current_layer_idx:
    #     #     decay_value = decay_from_first(entry_layer_idx, L)
    #     # else:
    #     #     decay_value = decay_from_last(entry_layer_idx, L)
        
    #     # if entry.layer_idx < num_encoder_layers:
    #     #     decay_value *= 0.1
    #     #     print(entry.layer_idx, entry.expert_idx, entry.r, decay_value)
    #     decay_value = decay_from_first(entry_layer_idx, L)

    #     # print(entry.layer_idx, entry.expert_idx, entry.r, decay_value)
    #     # topo_expert_score[entry.layer_idx, entry.expert_idx] = entry.r * decay_value
    #     topo_expert_score[entry.layer_idx, entry.expert_idx] = decay_value
    # # print(topo_expert_score)
    topo_expert_score = topo_expert_score / np.sum(topo_expert_score) + 1e-6
    # print("topo_expert_score", np.sum(topo_expert_score), np.max(topo_expert_score), np.min(topo_expert_score), topo_expert_score.shape)

    seq_expert_score = np.zeros_like(decoder_entry.matrix)
    freq_sum = 0
    # zero_access = False
    for entry in trace_entries:
        freq_sum += entry.access
    
    if freq_sum == 0:
        # freq_sum = len(trace_entries)
        seq_expert_score = np.ones_like(seq_expert_score)
        freq_sum = 1
        zero_access = True
    else: 
        for entry in trace_entries:
            matrix = copy.deepcopy(entry.matrix)
            matrix[matrix > 0] = 1
            # q = (1 if zero_access else entry.access) / freq_sum
            # # matrix = matrix * q
            seq_expert_score += entry.matrix

    seq_expert_score[num_encoder_layers:, :] = 0

    seq_expert_score = seq_expert_score / np.sum(seq_expert_score) + 1e-6
    # print("seq_expert_score", np.sum(seq_expert_score), np.max(seq_expert_score), np.min(seq_expert_score), seq_expert_score.shape)

    # decoder_matrix = np.zeros_like(decoder_entry.matrix)
    # if decoder_entry.num_new_tokens > 0:
    #     decoder_matrix = decoder_entry.matrix / decoder_entry.num_new_tokens
    # decoder_matrix = decoder_matrix + 1e-6
    decoder_matrix = decoder_entry.matrix
    if np.sum(decoder_matrix) == 0:
        decoder_matrix = np.ones_like(decoder_matrix)
    for i in range(decoder_matrix.shape[0]):
        if np.sum(decoder_matrix[i, :]) == 0:
            decoder_matrix[i, :] = 1
        decoder_matrix[i, :] = decoder_matrix[i, :] / np.sum(decoder_matrix[i, :])
    decoder_matrix = decoder_matrix / np.sum(decoder_matrix) + 1e-6
    # print("decoder_matrix", np.sum(decoder_matrix), np.max(decoder_matrix), np.min(decoder_matrix), decoder_matrix.shape)

    # total_score = topo_expert_score * seq_expert_score * decoder_matrix * frequency_score
    # total_score = topo_expert_score * seq_expert_score * decoder_matrix
    total_score = topo_expert_score * decoder_matrix * frequency_score
    # total_score = seq_expert_score * decoder_matrix
    # total_score = topo_expert_score * seq_expert_score 
    # total_score = topo_expert_score
    # total_score = topo_expert_score * (decoder_matrix + seq_expert_score)
    
    return convert_score_matrix_to_list(total_score)

    # # topo_score = topology_score(score_list, current_layer, total_layer)
    # seq_score = cross_sequence_score(score_list)
    # iter_score = cross_iteration_score(score_list)

    # return (1+topo_score) * (1+seq_score) * (1+iter_score)

