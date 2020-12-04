import pandas as pd
import panel as pn
pn.extension('plotly')
import plotly.express as px
import param
import numpy as np
import sys
sys.path.append(r'C:\Users\user\Google Drive\my projects\DS dashboard tool')
import utils


class BasicDashboardComponents(param.Parameterized):
    # variables to be displayed by widgets and set by the user:
    # select data column for scatter plot x-axis:
    X = param.ObjectSelector(default=None, objects=[], label='x axis data')
    # select data column for scatter plot y-axis:
    Y = param.ObjectSelector(default=None, objects=[], label='y axis data')
    # select column by which to filter the dataframe:
    cutoff_by_column = param.ObjectSelector(default=None, objects=[])
    # set cutoff value by which to filter (keeps rows with value >=cutoff_value:
    cutoff_value = param.Number(default=0, bounds=(0, 120), allow_None=True)
    # select column for boxplot view:
    var_to_inspect = param.ObjectSelector(default=None, objects=[])

    def __init__(self, df, *args, **kwargs):
        self.df = df
        super(type(self), self).__init__(*args, **kwargs)
        # update parameters values based on instance external inputs:
        self.X = self.df.columns.tolist()[0]  # set the default value of the widget
        self.param.X.objects = self.df.columns.tolist()  # set the list of objects to select from in the widget
        self.Y = self.df.columns.tolist()[1]
        self.param.Y.objects = self.df.columns.tolist()
        self.cutoff_by_column = self.df.columns.tolist()[0]
        self.param.cutoff_by_column.objects = self.df.columns.tolist()
        self.var_to_inspect = self.df.columns.tolist()[0]
        self.param.var_to_inspect.objects = self.df.columns.tolist()


    def filter_data_rows(self):
        '''
        filter the dataframe rows by a cutoff_value and a column set by the user.
        :return: dataframe where the values in column "cutoff_by_column" >= "cutoff_value"
        '''
        data = self.df.copy()
        data = utils.filter_df_rows(df=data, cutoff_by_column=self.cutoff_by_column, cutoff_value=self.cutoff_value)
        return data

    def metrics_table_view(self):
        '''
        :return: dataframe of regression performance metrics (mape, mdape, mse, rmse, R2) wrapped by Panel package
        '''
        data = self.filter_data_rows()
        metrics_dict = utils.calc_metrics_regression(data['actual'].to_numpy(), data['predicted'].to_numpy())
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').T
        return pn.pane.DataFrame(metrics_df, width=1350)

    def scatter_view(self):
        '''
        create a scatter plot using the filtered dataframe (output of self.filter_data_rows) and where x and y axes are
        dataframe columns set by the user via self.x and self.y
        :return: a Plotly scatter plot wrapped by Panel package
        '''
        data = self.filter_data_rows()
        array_x = data[self.X].to_numpy()
        array_y = data[self.Y].to_numpy()
        fig = utils.plot_scatter(x=array_x, y=array_y,
                                 add_unit_line=True, add_R2=True,
                                 layout_kwargs=dict(title='', xaxis_title=self.X, yaxis_title=self.Y))
        return pn.pane.Plotly(fig)  # pn.pane.Plotly is the Panel wrapper for Plotly figures


    def error_boxplot_view(self):
        '''
        create a boxplot using the filtered dataframe (output of self.filter_data_rows) and where the data is
        dataframe column set by the user via self.var_to_inspect.
        :return: a Plotly boxplot wrapped by Panel package
        '''
        data = self.filter_data_rows()
        fig = px.box(data, y=self.var_to_inspect, color_discrete_sequence=['green'])
        return pn.pane.Plotly(fig, width=400)  # pn.pane.Plotly is the Panel wrapper for Plotly figures

    def error_summary_stats_table(self):
        data = self.filter_data_rows()
        stats_df = data.describe(percentiles=[0.025, 0.2, 0.3, 0.5, 0.7, 0.8, 0.975]).T
        stats_df = stats_df.round(3)
        return pn.pane.DataFrame(stats_df, width=1350)


    def wrap_param_var_to_inspect(self):
        '''
        change the default widget of self.var_to_inspect from drop-down menu to Radio-button.
        :return: a widget wrapped by Panel package
        '''
        return pn.Param(self.param, parameters=['var_to_inspect'],  # pn.Param is a wrapper of Panel package for Param objects.
                 name='Choose variable to inspect distribution',
                 show_name=True,
                 widgets={'var_to_inspect': {'type': pn.widgets.RadioButtonGroup}},
                 width=200)


    @param.depends('cutoff_by_column', watch=True)
    def _update_cutoff_value(self):
        '''
        an update method to dynamically change the range of self.cutoff_value based on the user selection of self.cutoff_by_column.
        This method is triggered each time the chosen 'cutoff_by_column' change
        '''
        self.param['cutoff_value'].bounds = (self.df[self.cutoff_by_column].min(), self.df[self.cutoff_by_column].max())
        self.cutoff_value = self.df[self.cutoff_by_column].min()


def main():
    # For this demo, we generate synthetic data:
    d = utils.generate_synth_actual_and_predicted(N=200, mu=50, sigma=15, err_mu=0.5, err_sigma=1, err_scaling_factor=0.2)
    df_error = utils.make_error_df(actual=d['actual'], predicted=d['predicted'])
    #input dataframe to the dashboard:
    df_dashboard = df_error.copy()

    dash = BasicDashboardComponents(df_dashboard)

    # putting together all dashboard components
    # 1. wrapping dashboard components in Panel Layout objects (pn.Row or pn.Column):
    widgets_panel = pn.Column(dash.param['X'],
                              dash.param['Y'],
                              dash.param['cutoff_by_column'],
                              dash.param['cutoff_value'], width=200)

    scatter_panel = pn.Column(pn.Spacer(height=50),
                              dash.scatter_view)
    boxplot_panel = pn.Column(dash.wrap_param_var_to_inspect,
                              dash.error_boxplot_view)

    plots_panel = pn.Row(scatter_panel, boxplot_panel, width=1000, height=400) #width_policy='max',

    tables_panel = pn.Column(pn.pane.Markdown('## Model Performance Metrics', style={'font-family': "serif"}),
                             dash.metrics_table_view,
                             pn.pane.Markdown('## Descriptive Stats Table', style={'font-family': "serif"}),
                             dash.error_summary_stats_table)
    # 2. assembling all layout components into one:
    dashboard = pn.Column(pn.pane.Markdown('# Error Analysis Dashboard', style={'font-family': "serif"}),
                          pn.Row(widgets_panel, pn.Spacer(width=20), plots_panel),
                          pn.Spacer(height=50),
                                tables_panel)



    dashboard.show()

if __name__ == '__main__':
    main()