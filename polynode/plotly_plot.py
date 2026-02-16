import multiprocessing as mp
import time

import logging
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import torch
from dash import dcc, html
import flask

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

shutdown_flag = mp.Value('b', False)

# Dash app runner functions for each plot type
def dash_app_worker(shared_data, plot_type='data', kwargs=None):
    import dash
    from dash import dcc, html
    import plotly.graph_objs as go
    app = dash.Dash(__name__)
    if kwargs is None:
        kwargs = {}

    def get_figs():
        figs = []
        for key, value in shared_data.items():
            # Only display if value is a Plotly figure
            #print(f"Key: {key}, Value: {value}")
            if hasattr(value, 'to_plotly_json'):
                figs.append((key, value))
        return figs

    app.layout = html.Div([
        html.H2(kwargs.get("title", "Live Plots")),
        dcc.Interval(id='interval-component', interval=2000, n_intervals=0),
        html.Button('Terminate', id='terminate-button', n_clicks=0),
        html.Div(id='plots-container')
    ])

    # Store last seen timestamp
    app._last_timestamp = None


    def make_figure_div(pt, fig, idx):
        slider_id = f'size-slider-{idx}'
        graph_id = f'graph-{idx}'
        default_height = 600
        return html.Div([
            html.H4(f"{pt} plot"),
            dcc.Slider(id=slider_id, min=300, max=1200, step=10, value=default_height,
                       marks={300: '300px', 600: '600px', 900: '900px', 1200: '1200px'}),
            dcc.Graph(id=graph_id, figure=fig, style={'height': f'{default_height}px', 'width': '100%'})
        ], style={'margin-bottom': '40px'})

    app._last_timestamp = None

    @app.callback(
        dash.dependencies.Output('plots-container', 'children'),
        dash.dependencies.Input('interval-component', 'n_intervals')
    )

    def update_plots(n_intervals):
        timestamp = shared_data.get('timestamp', None)
        # Always rerender on initial page load (when n_intervals == 0), or if timestamp changes
        if n_intervals > 0 and app._last_timestamp is not None and timestamp == app._last_timestamp:
            raise dash.exceptions.PreventUpdate
        app._last_timestamp = timestamp
        figs = get_figs()
        return [make_figure_div(pt, fig, idx) for idx, (pt, fig) in enumerate(figs)]


    @app.callback(
        dash.dependencies.Output('terminate-button', 'disabled'),
        dash.dependencies.Input('terminate-button', 'n_clicks')
    )   

    def terminate_app(n_clicks):
        if n_clicks > 0:
            shutdown_flag.value = True
            func = flask.request.environ.get('werkzeug.server.shutdown')
            if func:
                func()
        return False
    
    # Callbacks for each slider to update the corresponding figure size
    for idx in range(20):  # Support up to 20 figures
        slider_id = f'size-slider-{idx}'
        graph_id = f'graph-{idx}'
        @app.callback(
            dash.dependencies.Output(graph_id, 'style'),
            [dash.dependencies.Input(slider_id, 'value')],
        )
        def update_graph_size(height, graph_id=graph_id):
            return {'height': f'{height}px', 'width': '100%'}

    app.run_server(debug=False, host="0.0.0.0", port=kwargs.get("port", 8050))

def start_dash_app_multiprocessing(plot_type='data', kwargs=None):
    """
    Starts Dash app in a separate process. Returns shared_data dict and process handle.
    Update shared_data['data'] in your training loop to update the plot.
    """
    manager = mp.Manager()
    shared_data = manager.dict()
    shared_data['timestamp'] = time.time()
    p = mp.Process(target=dash_app_worker, args=(shared_data, plot_type, kwargs))
    p.start()
    return shared_data, p

def plot_trajectories(trajectories, labels=None, title="Trajectories", target_points=None, target_labels=None):
    """
    Plot 2D or 3D trajectories using Plotly, including points and optional target points.

    Parameters:
        trajectories (torch.Tensor): Tensor of shape (timepoints, batchsize, 2) or (timepoints, batchsize, 3).
        labels (torch.Tensor or list of str, optional): Labels for each trajectory in the batch. Used to set the color.
        title (str, optional): Title of the plot.
        target_points (torch.Tensor or np.ndarray, optional): Array of shape (n_labels, 2) or (n_labels, 3) for target points.
        target_labels (torch.Tensor or list of str, optional): Labels for the target points. Should match the unique trajectory labels.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure object.

    Raises:
        ValueError: If `target_points` and `target_labels` lengths do not match.
    """
    trajectories = trajectories.detach().cpu().numpy()  # Convert to NumPy array if it's a tensor
    timepoints, batchsize, dims = trajectories.shape

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy().tolist()  # Convert tensor labels to a list

    if target_points is not None:
        target_points = target_points.detach().cpu().numpy() if isinstance(target_points, torch.Tensor) else np.array(target_points)
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.detach().cpu().numpy().tolist()  # Convert tensor target labels to a list
        if target_labels is None or len(target_points) != len(target_labels):
            raise ValueError("`target_points` and `target_labels` must have the same length.")

    # Define a color palette for labels
    all_labels = set()
    if labels:
        all_labels.update(map(str, labels))
    if target_labels:
        all_labels.update(map(str, target_labels))
    color_palette = px.colors.qualitative.Set1
    label_to_color = {str(label): color_palette[i % len(color_palette)] for i, label in enumerate(sorted(all_labels))}

    fig = go.Figure()

    # Plot trajectories
    for i in range(batchsize):
        traj = trajectories[:, i, :]
        color = label_to_color[str(labels[i])] if labels else f'Trajectory {i+1}'

        if dims == 2:  # 2D trajectory
            fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='lines+markers', name=f'Trajectory {labels[i]}',
                                     line=dict(color=color), marker=dict(size=1)))
        elif dims == 3:  # 3D trajectory
            fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines+markers', name=f'Trajectory {labels[i]}',
                                        line=dict(color=color), marker=dict(size=1)))

    # Plot target points
    if target_points is not None:
        for target, target_label in zip(target_points, target_labels):
            color = label_to_color[str(target_label)]
            if dims == 2:  # 2D target point
                fig.add_trace(go.Scatter(x=[target[0]], y=[target[1]], mode='markers', name=f'Target {target_label}',
                                         marker=dict(color=color, size=10, symbol='diamond')))
            elif dims == 3:  # 3D target point
                fig.add_trace(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], mode='markers', name=f'Target {target_label}',
                                            marker=dict(color=color, size=10, symbol='diamond')))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y", scene=dict(zaxis_title="Z"))
    return fig

def plot_loss(losses, epochs=None, title="Training Loss", log_scale=False):
    """
    Plot training loss over epochs using Plotly.

    Parameters:
        losses (list or np.ndarray): List of loss values.
        epochs (list or np.ndarray, optional): List of epoch numbers. If None, use range(len(losses)).
        title (str, optional): Title of the plot.
        log_scale (bool, optional): Whether to plot the loss on a logarithmic scale.

    Returns:
        plotly.graph_objs.Figure: The Plotly figure object.
    """
    if epochs is None:
        epochs = np.arange(len(losses))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=losses, mode='lines+markers', name='Loss', line=dict(color='blue')))

    fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Loss")

    if log_scale:
        fig.update_layout(yaxis_type="log")

    return fig

