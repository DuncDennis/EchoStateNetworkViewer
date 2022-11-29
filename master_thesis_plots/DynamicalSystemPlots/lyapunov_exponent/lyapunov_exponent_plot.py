import numpy as np
import plotly.graph_objects as go
import src.esn_src.simulations as sims
import src.esn_src.measures as meas

dt = 0.05
sys_obj = sims.Lorenz63(dt=dt)
systemname="Lorenz"


iterator_func = sys_obj.iterate
starting_point = sys_obj.default_starting_point

lle_array = meas.largest_lyapunov_exponent(
    iterator_func,
    starting_point=starting_point,
    deviation_scale=10 ** -10,
    steps=int(0.5*10**3),
    part_time_steps=15,
    steps_skip=50,
    dt=dt,
    return_convergence=True
)


height = 500
width = int(1.4 * height)
font_size = 16
font_size = 60
font_family = "Times New Roman"
legend_font_size = 16

xaxis_title = r"$\text{renormalisation step } i$" # N_\text{r}
# xaxis_title = r"Non-Latext test" # N_\text{r}
# yaxis_title =  r'$\text{running average of LLE } \lambda_\text{avg}$'
yaxis_title =  r'$\text{Largest lyapunov exponent } \lambda$'
yaxis_title =  r'Test'
yaxis_title =  r'$\huge{Test}$'

fig = go.Figure()

y = lle_array
x = np.arange(1, y.size + 1)
fig.add_trace(
    go.Scatter(x=x,
               y=y,
               name=r"$\text{running average }\lambda_\text{avg}(i)$",
               line=dict(width=3))
)

lle = y[-1].round(3)
# fig.add_hline(y=lle,
#               line=dict(
#                   color="black",
#                   dash="dot"
#               ),
#               name=fr"${lle}$",
#               )
name = r"$\text{LLE }\lambda = "
name = name + rf"{lle}$"
print(name)
fig.add_trace(
    go.Scatter(
        x=[0, x[-1]],
        y=[lle, lle],
        line=dict(
            color="black",
            dash="dot",
            width=4
        ),
        name=name,
        mode="lines"
    )
)

fig.update_layout(
    width=width,
    height=height,
    yaxis_title=yaxis_title,
    xaxis_title=xaxis_title,
    font=dict(
        size=font_size,
        family=font_family
    ),
    legend=dict(
        # orientation="h",
        yanchor="bottom",
        # y=1.01,  # 0.99
        # y=0.97,  # 0.99
        y=0.7,  # 0.99
        xanchor="left",
        # x=0.69,
        x=0.6,
        # font=dict(size=legend_font_size)
    ),
    # title=dict(font=dict(size=30)),
    # template="plotly_white",
    template="simple_white",
    showlegend=True,
    margin=dict(l=20, r=20, t=20, b=20),
)

fig.update_xaxes(range=[0, y.size+1],
                 title_font_size=30
                 )
# fig.update_yaxes(title_font_size=35)

file_name = f"lle_convergence_{systemname}.pdf"
fig.write_image(file_name, scale=3, format="pdf")

file_name = f"lle_convergence_{systemname}.png"
fig.write_image(file_name, scale=3, format="png")

# file_name = f"lle_convergence_{systemname}.svg"
# fig.write_image(file_name, scale=3, format="svg")


# https://stackoverflow.com/questions/5835795/generating-pdfs-from-svg-input
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF
# drawing = svg2rlg(file_name)
# renderPDF.drawToFile(drawing, "file.pdf")
