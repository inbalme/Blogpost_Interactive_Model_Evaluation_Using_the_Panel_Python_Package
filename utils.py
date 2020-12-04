import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score


def make_error_df(actual, predicted):
    '''
    returns a dataframe with prediction errors (diff-error and relative-errors)
    from input numpy arrays of actual and predicted values.
    :param actual: numpy array
    :param predicted: numpy array
    :return:
    '''
    dict = {'actual': actual, 'predicted': predicted}
    error_df = pd.DataFrame(data=dict)
    error_df['error'] = error_df['actual'] - error_df['predicted']
    error_df['error_relative_to_predicted'] = error_df['error'] / error_df['predicted']
    error_df['error_relative_to_actual'] = error_df['error'] / error_df['actual']
    return error_df


def plot_scatter(x, y, add_unit_line=True, add_R2=True, layout_kwargs=None):
    '''
    makes a scatter plot with Plotly
    :param x: numpy array
    :param y: numpy array
    :param add_unit_line: boolean. whether to add a line x=y
    :param add_R2: boolean. whether to add text of R2 on the plot
    :param layout_kwargs: a dictionary, according to plotly API:
        https://plotly.com/python/figure-labels/#manual-labelling-with-graph-objects
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html?highlight=update%20layout#plotly.graph_objects.Figure.update_layout
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html
        for example:
        1. layout_kwargs = dict(title='scatter', xaxis_title='X', legend_title='legend', showlegend=True)
        2. layout_kwargs = dict(title=dict(text='scatter'), xaxis=dict(title={'text': 'X'}))
    :return: plotly figure object
    '''
    # need to add titles, colors, fonts, (in **kwargs), option to add more traces (perhaps input should be tuples of [(x1,y1), (x2,y2)...])
    #add layout_kwargs for 'showlegend', 'title', 'xaxis_title', 'legend_title', 'fonts', etc.)
    unit_line = [np.nanmin(np.append(x, y)), np.nanmax(np.append(x, y))]
    # calculate explained variance:
    from sklearn.metrics import explained_variance_score
    R2 = round(explained_variance_score(x, y), 3)
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='markers',
                             name='',
                            #  marker=dict(
                            #     color=color='rgb(139,0,139)',
                            #     colorscale='Viridis',
                            #     line_width=None
                            # ) #accept input of marker_kwargs
    ))

    if add_R2:
        fig.add_annotation(
            x=unit_line[0],
            y=unit_line[1],
            showarrow=False,
            text=f'R2 score is: {R2}')
    # fig.add_trace(go.Scatter(x=random_x, y=random_y1,
    #                          mode='lines+markers',
    #                          name='lines+markers'))
    if add_unit_line:
        fig.add_trace(go.Scatter(x=unit_line, y=unit_line,
                                 mode='lines',
                                 name=''))
    fig.update_layout(showlegend=False)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    # Set options common to all traces with fig.update_traces
    # fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    # fig.update_layout(title='Styled Scatter',
    #                   yaxis_zeroline=False, xaxis_zeroline=False)
    return fig


# layout_kwargs = dict(title=dict(text='scatter'), xaxis=dict(title={'text': 'X'})) #dict(showlegend=True) #dict(title='scatter', xaxis_title='X', legend_title='legend', showlegend=True)
# fig = plot_scatter(x=d['actual'], y=d['predicted'], layout_kwargs=layout_kwargs)




def calc_metrics_regression(actual, predicted, digits=3):
    '''
    returns a dictionary of regression metrics calculated from input numpy arrays of actual and predicted values
    :param actual:
    :param predicted:
    :param digits:
    :return:
    '''
    R2_score = round(explained_variance_score(actual, predicted), digits)
    mae = round(np.nanmean(np.abs(predicted - actual)), digits)
    mse = round(np.nanmean((predicted - actual) ** 2), digits)
    rmse = round(np.nanmean((predicted - actual) ** 2) ** .5, digits)
    corr = round(np.corrcoef(predicted, actual)[0, 1], digits)
    mape = round(np.nanmean(np.abs(predicted - actual) / np.abs(actual)), digits)
    mdape = round(np.median(np.abs(predicted - actual) / np.abs(actual)), digits)
    return({'mape': mape, 'mdape': mdape, 'mae': mae,
            'mse': mse, 'rmse': rmse,
            'corr': corr, 'R2_score': R2_score})


def filter_df_rows(df, cutoff_by_column=None, cutoff_value=None):
    '''
    returns a filtered dataframe where values in column "cutoff_by_column" >= "cutoff_value"
    :param df:
    :param cutoff_by_column:
    :param cutoff_value:
    :return:
    '''
    # need to verify that if cutoff_by_column or cutoff_value are None then the function returns the dataframe intact.
    if cutoff_by_column is None:
        cutoff_by_column = df.columns.to_list()[0]
    if cutoff_value is None:
        # cutoff_value = np.min(df[cutoff_by_column])
        return df
    else:
        return df[df[cutoff_by_column] >= cutoff_value]

def generate_synth_actual_and_predicted(N=100, mu=0, sigma=1, err_mu=0, err_sigma=1, err_scaling_factor=1, seed=1):
    '''
    generates a numpy array of "actual" values: N random numbers from a normal distribution with mean=mu and std=sigma
    and a numpy array of "predicted" values: the "actual" values with an additive or multiplicative error.
    :param N: length of generated vector of actual and predicted values
    :param mu: mean of normal distribution that generates the actual values
    :param sigma: std of normal distribution that generates the actual values
    :param err_mu: mean of normal distribution that generates the error values
    :param err_sigma: std of normal distribution that generates the error values
    :param err_scaling_factor: scale of the error
    :return: a dictionary with keys: 'actual', 'predicted'
    '''
    np.random.seed(seed)
    actual = np.random.normal(loc=mu, scale=sigma, size=N)
    predicted = actual + err_scaling_factor * actual * np.random.normal(loc=err_mu, scale=err_sigma, size=N)
    return {'predicted': predicted, 'actual': actual}


def main():
    d = generate_synth_actual_and_predicted(N=200, mu=50, sigma=15, err_mu=0.5, err_sigma=1, err_scaling_factor=0.2)
    df_error = make_error_df(actual=d['actual'], predicted=d['predicted'])
    out = filter_df_rows(df_error, cutoff_by_column=df_error.columns.to_list()[1], cutoff_value=0)
    print(out.head())


if __name__ == '__main__':
    main()
