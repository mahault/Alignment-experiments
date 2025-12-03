

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import ipywidgets as widgets
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML


# default function to convert actions to string representation
# can be overridden by passing a custom function to the `plot_tree` function
def action_to_string(action, model=None):
    if model is None:
        return str(action)
    
    # we want to check first two elements of the action array
    action_parts = []
    
    for i in range(min(2, len(model.B), len(action))):
        action_idx = action[i]
        k = list(model.B[i].batch.keys())[-1]  # the last key is the control key
        actions_list = model.B[i].batch[k]
        
        if action_idx < len(actions_list):
            action_name = actions_list[action_idx]
            # only add non-noop actions to the description
            if action_name != "noop":
                action_parts.append(action_name)
    
    # if no named actions [basically [0,0]], return "noop"
    if not action_parts:
        return "noop"
    
    return ", ".join(action_parts)


def observation_to_string(observation, model=None):
    if model is None:
        return str(observation[0])
    
    obs = ""
    for i in range(len(model.A)):
        for k,v in model.A[i].event.items():
            # v should have string values for each observation modality
            obs_str = str(v[observation[0, i]])
            obs += obs_str
            obs += ",\n"

    return obs


def formatting_jax(value, format_str=".2f"):
    try:
        if hasattr(value, "shape"):
            if value.shape == ():
                return f"{float(value):{format_str}}"
            elif len(value.shape) == 1 and value.shape[0] == 1:
                return f"{float(value[0]):{format_str}}"
            elif len(value.shape) == 1 and value.shape[0] <= 4:
                return (
                    "[" + ", ".join([f"{float(x):{format_str}}" for x in value]) + "]"
                )
            else:
                return (
                    "[" + ", ".join([f"{float(x):{format_str}}" for x in value]) + "]"
                )
        else:
            return f"{float(value):{format_str}}"
    except:
        return str(value)[:10]


def plot_plan_tree(
    tree,
    model=None,
    root_node=None,
    max_depth=15,
    min_prob=0.2,
    observation_description=observation_to_string,
    action_description=action_to_string,
    figsize=(15, 15),
    font_size=8,
    node_size=500,
    layout="dot",
    ax=None,
):

    G = nx.DiGraph()
    node_labels = {}
    node_colors = []

    policy_cmap = plt.cm.Purples  # cool colours for non-tom agent policies
    obs_cmap = plt.cm.Oranges  # cool colours for non-tom agent observations

    if root_node is None:
        root_node = tree.root()

    queue = [(root_node, None, 0)]  # (node, parent_id, depth)
    node_id = 0
    nodes_processed = 0

    while queue:
        current, parent, depth = queue.pop(0)
        nodes_processed += 1
        if depth > max_depth:
            continue

        label_parts = [str(current["idx"])]

        color_map = plt.cm.Greys
        color = to_rgba("lightgrey")

        # observation nodes
        if "observation" in current and current["idx"] != tree.root()["idx"]:
            label_parts.append(f"{observation_description(current['observation'], model)}")

            if "prob" in current:
                label_parts.append(f"P:{formatting_jax(current['prob'])}")

            # add G_recur for observation nodes
            if "G_recursive" in current:
                label_parts.append(f"G:{formatting_jax(current['G_recursive'][0])}")
            else:
                label_parts.append(f"G:{formatting_jax(current['G'])}")

            color_map = obs_cmap

        # focal agent policy nodes
        elif "policy" in current:
            label_parts.append(f"{action_description(current['policy'], model)}")

            if "prob" in current:
                label_parts.append(f"P:{formatting_jax(current['prob'])}")

            if "G" in current:
                label_parts.append(f"G:{formatting_jax(current['G'])}")

            color_map = policy_cmap

        if "agent" in current:
            # if this is an observation node for a specific agent, use a different color
            colors = [0.3, 0.5, 0.7]
            color = color_map(colors[int(current["agent"])])
        else:
            color = color_map(0.5)

        node_colors.append(color)

        G.add_node(
            node_id,
            idx=current["idx"],
            type="policy" if "policy" in current else "observation",
        )
        node_labels[node_id] = "\n".join(label_parts)

        if parent is not None:
            G.add_edge(parent, node_id)

        # check for appropriate minimum probability based on node type
        for i, child_idx in enumerate(current.get("children", [])):
            child = tree[child_idx]
            prob = current["children_probs"][i]
            skip_child = False

            if prob < min_prob:
                skip_child = True
            else:
                child["prob"] = prob  # add probability to child node

            if not skip_child:
                queue.append((child, node_id, depth + 1))

        node_id += 1

        if nodes_processed > 1000:
            print(f"Warning: visualisation is limited to first 1000 nodes")
            break

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    try:
        if layout == "dot":  # traditional hierarchical
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        elif layout == "twopi":  # radial
            pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
        else:
            pos = nx.spring_layout(G, k=0.3, iterations=50)  # default fallback
    except Exception as e:
        print(f"Layout error: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(G, k=0.3, iterations=50)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, arrows=True, arrowsize=5, ax=ax)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.8,
        linewidths=1,
        edgecolors="gray",
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=font_size,
        verticalalignment="center",
        horizontalalignment="center",
        ax=ax,
    )

    ax.set_axis_off()

    legend_elements = [
        Patch(
            facecolor=policy_cmap(0.5),
            edgecolor="gray",
            alpha=0.8,
            label="Policy Nodes",
        ),
        Patch(
            facecolor=obs_cmap(0.5),
            edgecolor="gray",
            alpha=0.8,
            label="Observation Nodes",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    return G, pos


def visualize_plan_tree(
    info, 
    time_idx=0,
    agent_idx=0,
    plotting_other_intom = False,
    root_idx=None,
    model=None,
    observation_description=observation_to_string,
    action_description=action_to_string,
    depth=4,
    min_prob=0.2,
    fig_size=(15, 15)
):
    if plotting_other_intom:
        tree = jtu.tree_map(lambda x: x[agent_idx, time_idx], info["other_tree"])
    else: 
        tree = jtu.tree_map(lambda x: x[agent_idx, time_idx], info["tree"])
        
    depth_slider = widgets.IntSlider(
        value=depth,
        min=1,
        max=15,
        step=1,
        description="Max Depth:",
        continuous_update=False,
    )

    prob_slider = widgets.FloatSlider(
        value=min_prob,
        min=0.0,
        max=0.5,
        step=0.05,
        description="Min Probability:",
        continuous_update=False,
    )

    layout_dropdown = widgets.Dropdown(
        options=[
            ("Hierarchical (dot)", "dot"),
            ("Radial (twopi)", "twopi"),
        ],
        value="dot",
        description="Layout:",
    )

    node_size_slider = widgets.IntSlider(
        value=500,
        min=200,
        max=1000,
        step=50,
        description="Node Size:",
        continuous_update=False,
    )

    font_size_slider = widgets.IntSlider(
        value=8,
        min=6,
        max=16,
        step=1,
        description="Font Size:",
        continuous_update=False,
    )

    # add a node selector capability
    def get_all_nodes(tree):
        used_indices = jnp.argwhere(tree.used[:, 0])[:, 0]

        root = tree.root()
        nodes = [(root, root["idx"], "Root")]  # (node, index, description)
        node_options = [("Root", root["idx"])]

        for idx in used_indices:
            nodes.append((None, idx, ""))
            node_options.append((f"{idx}", idx))

        return nodes, node_options

    _, node_options = get_all_nodes(tree)

    node_dropdown = widgets.Dropdown(
        options=node_options,
        value=tree.root()["idx"] if root_idx is None else root_idx,
        description="Focus Node:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="50%"),
    )

    def find_node_by_idx(tree, target_idx):
        return tree[target_idx]

    fig, ax = plt.subplots(figsize=fig_size)
    output = widgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            ax.clear()

            # get the selected node using our improved lookup
            selected_node_idx = node_dropdown.value
            selected_node = find_node_by_idx(tree, selected_node_idx)

            def get_nearest_node(event):
                min_dist = float("inf")
                closest_node = None
                for node, (x, y) in pos.items():
                    dist = (event.xdata - x) ** 2 + (event.ydata - y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = node
                return closest_node

            # Click event handler
            def on_click(event):
                if event.inaxes:
                    node = get_nearest_node(event)
                    with output:
                        node_dropdown.value = int(G.nodes[node]["idx"])
                        update_plot()

            G, pos = plot_plan_tree(
                tree,
                model=model,
                root_node=selected_node,
                max_depth=depth_slider.value,
                min_prob=prob_slider.value,
                observation_description=observation_description,
                action_description=action_description,
                figsize=fig_size,
                node_size=node_size_slider.value,
                font_size=font_size_slider.value,
                layout=layout_dropdown.value,
                ax=ax,
            )
            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.tight_layout()
            fig.canvas.draw_idle()

    depth_slider.observe(update_plot, names="value")
    prob_slider.observe(update_plot, names="value")
    layout_dropdown.observe(update_plot, names="value")
    node_size_slider.observe(update_plot, names="value")
    font_size_slider.observe(update_plot, names="value")
    node_dropdown.observe(update_plot, names="value")

    ui = widgets.VBox(
        [
            widgets.HBox([depth_slider, prob_slider]),
            widgets.HBox([layout_dropdown, node_size_slider, font_size_slider]),
            widgets.HBox(
                [
                    node_dropdown,
                    widgets.Button(description="Update Plot", on_click=update_plot),
                ]
            ),
            output,
        ]
    )

    update_plot()
    return ui


def visualize_beliefs(info, agent_idx=0, model=None):
    """Plot the results of the agent's beliefs and actions."""
    num_plots = len(info["qs"])
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 2*num_plots), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Assuming all qs[i] have the same width along axis=2
    x_len = info["qs"][0].shape[1]
    x_ticks = np.arange(x_len + 1)

    for i, ax in enumerate(axes):
        if model is not None:
            title = list(model.B[i].event.keys())[0]
        else:
            title = f"state factor {i}"
        ax.set_title(title, fontsize=10)

        # Plot object location beliefs as greyscale intensity
        ax.imshow(info["qs"][i][agent_idx, :, :].T, cmap="gray_r", vmin=0.0, vmax=1.0, aspect='auto')

        ax.set_yticks(jnp.arange(info["qs"][i].shape[-1]))
        if model is not None:
            ax.set_yticklabels(model.B[i].event[list(model.B[i].event.keys())[0]])
        ax.set_xticks(x_ticks)

    # Only bottom subplot shows x-axis labels
    axes[-1].set_xlabel("Time step")
    fig.subplots_adjust(hspace=0.6)  # More vertical spacing
    plt.tight_layout()
    plt.show()


def visualize_env(
    info,
    model=None,
    observation_description=observation_to_string,
    action_description=action_to_string,
    save_as_gif=False,
    gif_filename="rollout.gif",
):
    try:
        batch_size = info["env"].num_agents
    except:
        batch_size = 1
    num_timesteps = info["qs"][0].shape[1]
    
    # adjust figure size based on number of agents - reduce height to eliminate bottom white space
    base_height = 3.5
    height_per_agent = 1.1
    fig_height = base_height + (batch_size - 1) * height_per_agent
    fig, ax = plt.subplots(figsize=(4, fig_height))
    
    def update(time_idx):
        ax.clear()
        ax.axis("off")
        ax.set_aspect("equal")

        # render the env
        env = jtu.tree_map(lambda x: x[:, time_idx], info["env"])
        ax.imshow(env.render())
        
        # prepare title string for both agents
        title_str = f"Timestep {time_idx}\n"

        agent_colours = ["red", "purple", "yellow"]
        
        # get the observations and actions for the current timestep
        for agent_idx in range(batch_size): 
            observation = jtu.tree_map(lambda x: x[agent_idx, time_idx][None, ...], info["observation"])
            observation = jnp.concatenate(observation, axis=-1)
            action = jtu.tree_map(lambda x: x[agent_idx, time_idx], info["action"])
            
            obs = observation_description(observation, model).replace("\n", " ")
            act = action_description(action, model).replace("\n", " ")

            agent_colour = agent_colours[agent_idx % len(agent_colours)]
            
            title_str += f"\n({agent_colour}) Agent {agent_idx} observed ({obs}) \n and selects action ({act}) \n"

        title_y = 1.2 + (batch_size - 1) * 0.125
        ax.text(0.5, title_y, title_str, fontsize=8, 
                ha='center', va='top', transform=ax.transAxes)

    # display the animation
    anim = FuncAnimation(fig, update, frames=num_timesteps, repeat=True, interval=1000)
    
    # the gif gets placed in the bottom of the figure, with the top reserved for title and text
    plt.subplots_adjust(bottom=0.05, top=0.8)

    plt.close(fig)
    display(HTML(anim.to_jshtml()))

    if save_as_gif:
        anim.save(gif_filename, writer="imagemagick", fps=1)