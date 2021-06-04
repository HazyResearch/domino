from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import HoverTool


#output_notebook()

source = ColumnDataSource(data=dict(
    x=[1, 2, 3, 4, 5],
    y=[2, 5, 8, 2, 7],
    desc=['A', 'b', 'C', 'd', 'E'],
    imgs=[
        '/Users/sabrieyuboglu/code/domino-21/notebooks/CF3C3ACB-229D-4C0F-935C-20C99BEEEFC0_1_105_c.jpeg',
    ] * 5,
    fonts=[
        '<i>italics</i>',
        '<pre>pre</pre>',
        '<b>bold</b>',
        '<small>small</small>',
        '<del>del</del>'
    ]
))



p = figure(plot_width=400, plot_height=400,
           title="Mouse over the dots")

hover = HoverTool(tooltips ="""
            <div>
                <div>
                    <img
                        src="file://@imgs" alt="@imgs" 
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 15px;">@f_count @f_interval</span>
                    <span style="font-size: 10px; color: #696;">($x, $y)</span>
                </div>
            """)


# Add the hover tool to the graph
p.add_tools(hover)

p.circle('x', 'y', size=20, source=source)

show(p);