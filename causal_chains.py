import pyphi
import networkx as nx

import numpy as np
import pyphi
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from itertools import product

def get_effects_from_source(
    transitions, source_nodes=None, max_purview_size=5, max_mechanism_size=5,
):

    if source_nodes == None:
        source_nodes = range(len(transitions[0].network))

    source_subsets = list(
        pyphi.utils.powerset(source_nodes, nonempty=True, max_size=max_mechanism_size)
    )

    all_purviews = list(
        pyphi.utils.powerset(
            range(len(transitions[0].network)), nonempty=True, max_size=max_purview_size
        )
    )

    past_effect_subsets = []
    effects = []

    # loop through all transitions
    for i, t in enumerate(tqdm(transitions, desc="Computing legal effects")):

        # compute the directed account for all valid mechansisms over all system subsets
        account = pyphi.actual.directed_account(
            t,
            pyphi.direction.Direction.EFFECT,
            mechanisms=set(past_effect_subsets + source_subsets),
            purviews=all_purviews,
            allow_neg=False,
        )
        # NOTE: Here negative alpha are not allowed (negative connectedness could still be possible depending on later definitions)

        # adding necessary variables to the growing list of effects
        effects.append(
            [
                [effect.mechanism, effect.purview, effect.alpha]
                for effect in account.irreducible_effects
            ]
        )

        # updating which sink_nodes subsets will be used as mechanisms in the next iteration.
        # only those that are completely contained in a an actual effect from this iteration.
        past_effect_subsets = [
            mechanism_subset
            for mechanism_subset in all_purviews
            if any(
                [
                    all([element in effect.purview for element in mechanism_subset])
                    for effect in account.irreducible_effects
                ]
            )
        ]
    return effects

def get_causes_of_sink(
    transitions, sink_nodes=None, max_purview_size=5, max_mechanism_size=5,
):

    if sink_nodes == None:
        sink_nodes = range(len(transitions[0].network))

    sink_subsets = list(
        pyphi.utils.powerset(sink_nodes, nonempty=True, max_size=max_mechanism_size)
    )

    all_subsets = list(
        pyphi.utils.powerset(
            range(len(transitions[0].network)),
            nonempty=True,
            max_size=max_mechanism_size,
        )
    )

    past_cause_subsets = []
    causes = []
    # loop through all transitions
    for i, t in enumerate(
        tqdm(
            reversed(transitions), total=len(transitions), desc="Computing legal causes"
        )
    ):
        # compute the directed account for all valid mechansisms over all system subsets
        account = pyphi.actual.directed_account(
            t,
            pyphi.direction.Direction.CAUSE,
            mechanisms=set(past_cause_subsets + sink_subsets),
            purviews=all_subsets,
            allow_neg=False,
        )
        # NOTE: Here negative alpha are not allowed (negative connectedness could still be possible depending on later definitions)

        # adding necessary variables to the growing list of effects
        causes.append(
            [
                [cause.mechanism, cause.purview, cause.alpha]
                for cause in account.irreducible_causes
            ]
        )

        # updating which sink_nodes subsets will be used as mechanisms in the next iteration.
        # only those that are completely contained in a an actual effect from this iteration.
        past_cause_subsets = [
            mechanism_subset
            for mechanism_subset in all_subsets
            if any(
                [
                    all([element in effect.purview for element in mechanism_subset])
                    for effect in account.irreducible_causes
                ]
            )
        ]

    return causes

def check_legal_chain(
    cause_effects,
    chain,
    source_subset,
    sink_subset,
    time_restricted_source,
    all_source_subsets,
    all_sink_subsets,
    direction="cause",
):

    mechanisms = [
        cause_effects[chain[i]][0] for i, cause_effects in enumerate(cause_effects)
    ]
    purviews = [
        cause_effects[chain[i]][1] for i, cause_effects in enumerate(cause_effects)
    ]

    if direction == "cause":

        if time_restricted_source and all_source_subsets:
            legal_source = all([element in source_subset for element in purviews[-1]])
        elif all_source_subsets:
            legal_source = any(
                [
                    all([element in source_subset for element in purview])
                    for purview in purviews
                ]
            )
        elif time_restricted_source:
            legal_source = purviews[0] == source_subset
        else:
            legal_source = any([purview == source_subset for purview in purviews])

        if all_sink_subsets:
            legal_sink = all([element in sink_subset for element in mechanisms[0]])
        else:
            legal_sink = mechanisms[0] == sink_subset

    else:  # for effects
        if time_restricted_source and all_source_subsets:
            legal_source = all([element in source_subset for element in mechanisms[0]])
        elif all_source_subsets:
            legal_source = any(
                [
                    all([element in source_subset for element in mechanism])
                    for mechanism in mechanisms
                ]
            )
        elif time_restricted_source:
            legal_source = mechanisms[0] == source_subset
        else:
            legal_source = any([mechanism == source_subset for mechanism in mechanisms])

        if all_sink_subsets:
            legal_sink = all([element in sink_subset for element in purviews[-1]])
        else:
            legal_sink = purviews[-1] == sink_subset

    return legal_source and legal_sink

def get_effect_chains(
    effects,
    start_t,
    source_subset,
    sink_subset,
    all_source_subsets=False,
    all_sink_subsets=False,
    time_restricted_source=False,
):

    relevant_effects = effects[start_t:]

    # list all potential chains through the system (based on causal links)
    potential_chains = [
        chain for chain in product(*[range(len(links)) for links in relevant_effects])
    ]
    effect_chains = []
    chain_alphas = []

    tau = len(potential_chains[0])

    # loop through every potential chain
    for chain in tqdm(potential_chains, desc="Computing effect chains"):
        possible = True

        # check that the first mechanism is completely in the input array
        mechanisms = [effect[chain[i]][0] for i, effect in enumerate(relevant_effects)]
        last_purview = relevant_effects[-1][chain[-1]][1]

        if check_legal_chain(
            relevant_effects,
            chain,
            source_subset,
            sink_subset,
            time_restricted_source,
            all_source_subsets,
            all_sink_subsets,
            direction="effect",
        ):

            for t in range(tau - 1):
                effect_purview = relevant_effects[t][chain[t]][1]
                next_mechanism = relevant_effects[t + 1][chain[t + 1]][0]

                # if not all elements in the next mechanism is in the current effect, the chain is broken
                if not all([element in effect_purview for element in next_mechanism]):
                    possible = False
                    break

            # store the possible chains with their respective alpha values
            if possible:
                effect_chain = [
                    (effect[link][0], effect[link][1], effect[link][2])
                    for effect, link in zip(relevant_effects, chain)
                ]

                # but keep only unique chains
                if not any([effect_chain == p for p in effect_chains]):
                    effect_chains.append(effect_chain)

    return effect_chains  # causal_bind

def get_cause_chains(
    causes,
    end_t,
    source_subset,
    sink_subset,
    all_source_subsets=False,
    all_sink_subsets=False,
    time_restricted_source=False,
):

    relevant_causes = causes[:end_t]

    # list all potential chains through the system (based on causal links)
    potential_chains = [
        chain for chain in product(*[range(len(links)) for links in relevant_causes])
    ]
    cause_chains = []
    chain_alphas = []

    tau = len(relevant_causes)

    potential = 0
    no_link = 0
    # loop through every potential chain
    for chain in tqdm(potential_chains, desc="Computing cause chains"):
        possible = True

        # check that the first mechanism is completely in the input array
        sink_mechanism = relevant_causes[0][chain[0]][0]
        purviews = [effect[chain[i]][1] for i, effect in enumerate(relevant_causes)]

        if check_legal_chain(
            relevant_causes,
            chain,
            source_subset,
            sink_subset,
            time_restricted_source,
            all_source_subsets,
            all_sink_subsets,
        ):
            for t in range(tau - 1):
                cause_purview = relevant_causes[t][chain[t]][1]
                next_mechanism = relevant_causes[t + 1][chain[t + 1]][0]

                # if not all elements in the next mechanism is in the current cause, the chain is broken
                if not all([element in cause_purview for element in next_mechanism]):
                    possible = False
                    break

            # store the possible chains with their respective alpha values
            if possible:
                cause_chain = [
                    (cause[link][0], cause[link][1], cause[link][2])
                    for cause, link in zip(relevant_causes, chain)
                ]

                # but keep only unique chains
                if not any([cause_chain == p for p in cause_chains]):
                    cause_chains.append(cause_chain)

    return cause_chains  # causal_bind

def get_bundled_chains(possible_chains):
    bundled_chains = (
        [
            list(set([chain[link] for chain in possible_chains]))
            for link in range(len(possible_chains[0]))
        ]
        if len(possible_chains) > 0
        else [[("NA", "NA", 0)]]
    )
    return bundled_chains

def get_bundle_strength(bundled_chains):
    return min([sum([link[2] for link in links]) for links in bundled_chains])



def get_system_causal_bundle(
    network,
    network_states,
    source_nodes,
    sink_nodes,
    max_purview_size=4,
    max_mechainsm_size=4,
    start_t=0,
    all_source_subsets=True,
    all_sink_subsets=True,
):

    transitions = [
        pyphi.actual.Transition(
            network,
            network_state_1,
            network_state_2,
            network.node_indices,
            network.node_indices,
        )
        for network_state_1, network_state_2 in zip(
            network_states[:-1], network_states[1:]
        )
    ]

    transitions = transitions[start_t:] if start_t > 0 else transitions

    effects = get_effects_from_source(
        transitions,
        source_nodes=source_nodes,
        max_purview_size=max_purview_size,
        max_mechanism_size=max_mechainsm_size,
    )
    
    causes = get_causes_of_sink(
        transitions,
        sink_nodes=sink_nodes,
        max_purview_size=max_purview_size,
        max_mechanism_size=max_mechainsm_size,
    )
    
    cause_chains = get_cause_chains(
        causes,
        len(transitions),
        source_nodes,
        sink_nodes,
        all_source_subsets=all_source_subsets,
        all_sink_subsets=all_sink_subsets,
    )
    
    effect_chains = get_effect_chains(
        effects,
        0,
        source_nodes,
        sink_nodes,
        all_source_subsets=all_source_subsets,
        all_sink_subsets=all_sink_subsets,
    )
    
    bundled_causes = get_bundled_chains(cause_chains)
    bundled_effects = get_bundled_chains(effect_chains)

    connectedness = sum(
        [get_bundle_strength(bundled_causes), get_bundle_strength(bundled_effects)]
    )

    return {
        "causes": bundled_causes,
        "effects": bundled_effects,
        "connectedness": connectedness,
    }

def draw_connectedness(
    network, source_nodes, sink_nodes, network_states, causal_bundle, title_prefix=""
):
    # define network
    G = nx.DiGraph()

    # define nodes
    node_states = []
    border_color = []
    node_position = dict()
    node_label = []
    for t in range(len(network_states)):
        for i in source_nodes + sink_nodes:
            node_name = network.node_labels[i] + " " + str(t)
            node_label.append(node_name)
            node_states.append("gray" if network_states[t][i] == 1 else "w")
            border_color.append("k" if i in sink_nodes else "b")
            node_position[node_name] = (t, i)
            G.add_node(node_name)

    # define edges
    edge_colors = []
    edge_styles = []
    edge_labels = []
    edge_widths = dict()
    counter = 0
    for direction, bundle in causal_bundle.items():
        if not direction == "connectedness":
            for t, links in enumerate(bundle):
                if direction == "causes":
                    for link in links:
                        if not link[0] == "NA":
                            mechanism = link[0]
                            purview = link[1]
                            alpha = link[2]

                            for m in mechanism:
                                for p in purview:
                                    source = (
                                        network.node_labels[m]
                                        + " "
                                        + str(len(network_states) - 1 - (t))
                                    )
                                    target = (
                                        network.node_labels[p]
                                        + " "
                                        + str(len(network_states) - 1 - (t + 1))
                                    )
                                    a = 0.11 + 2 * alpha

                                    if not (source, target) in edge_widths.keys():
                                        edge_colors.append("r")
                                        edge_widths[(source, target)] = a
                                        edge_labels.append((source, target))
                                    else:
                                        edge_widths[(source, target)] += a

                if direction == "effects":
                    for link in links:
                        if not link[0] == "NA":
                            mechanism = link[0]
                            purview = link[1]
                            alpha = link[2]

                            for m in mechanism:
                                for p in purview:
                                    source = network.node_labels[m] + " " + str((t))
                                    target = network.node_labels[p] + " " + str((t + 1))
                                    a = 0.11 + 2 * alpha

                                    if not (source, target) in edge_widths.keys():
                                        edge_colors.append("g")
                                        edge_widths[(source, target)] = a
                                        edge_labels.append((source, target))
                                    else:
                                        edge_widths[(source, target)] += a
    G.add_edges_from(edge_labels)

    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_nodes(
        G,
        node_position,
        node_color=node_states,
        node_size=500,
        node_shape="o",
        label=node_label,
        edgecolors=border_color,
        linewidths=3,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        node_position,
        edgelist=edge_labels,
        width=list(edge_widths.values()),
        edge_color=edge_colors,
        arrowstyle="-|>",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, node_position, {l: l[:-2] for l in node_label}, font_size=11, ax=ax
    )

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.title(
        title_prefix + "connectedness = {}".format(causal_bundle["connectedness"])
    )

    plt.xticks(
        ticks=list(range(len(network_states))),
        labels=[
            "t={}".format(-(len(network_states) - (t + 1)))
            for t in range(len(network_states))
        ],
    )
    plt.yticks(ticks=list(range(len(network))), labels=network.node_labels)
    plt.xlabel("time")
    plt.ylabel("nodes")
    return G

def get_all_subset_connectedness(
    network,
    network_states,
    source_nodes,
    sink_nodes,
    max_purview_size=4,
    max_mechainsm_size=4,
    start_t=0,
):
    for sink_subset in pyphi.utils.powerset(sink_nodes, nonempty=True):
        bundle = dict()
        max_source = ()
        connectedness = 0

        bundle = get_system_causal_bundle(
            network,
            network_states,
            source_nodes,
            sink_subset,
            max_purview_size=4,
            max_mechainsm_size=4,
            start_t=0,
            all_source_subsets=True,
            all_sink_subsets=False,
        )

        draw_connectedness(
            network,
            source_nodes,
            sink_nodes,
            network_states,
            bundle,
            title_prefix="{} ".format(sink_subset),
        )

    bundle = get_system_causal_bundle(
        network,
        network_states,
        source_nodes,
        sink_nodes,
        max_purview_size=4,
        max_mechainsm_size=4,
        start_t=0,
        all_source_subsets=True,
        all_sink_subsets=True,
    )
    draw_connectedness(
        network, source_nodes, sink_nodes, network_states, bundle, title_prefix="System "
    )