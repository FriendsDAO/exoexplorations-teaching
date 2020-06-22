import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly

from plotly.offline import init_notebook_mode

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
from plotly.offline import init_notebook_mode, iplot
from IPython.display import YouTubeVideo


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def lightcurve(planet_radius, star_radius, i, imsize=(400, 400)):
    area_star_fractional = []
    field = np.zeros(imsize)
    star = create_circular_mask(imsize[0], imsize[1], radius=star_radius)
    field[star] = 1.0
    area_star_total = np.sum(star)

    for x in np.arange(imsize[0]):
        planet = create_circular_mask(imsize[0], imsize[1], center=(x, imsize[1]/2+i), radius=planet_radius)
        field[star] = 1.0
        field[planet] = 0.0
        area_star_fractional.append(np.sum(field))
    
    area_star = np.array(area_star_fractional)/area_star_total
#     plt.imshow(star)

    return np.arange(imsize[0]), area_star

def show_transit_video():
    return YouTubeVideo('8v4SRfmoTuU', width=800, height=300)


def simulate(planet_radius, star_radius, inclination, imsize=(400, 400)):
    time, flux = lightcurve(planet_radius, star_radius, i=inclination, imsize=imsize)
    dfs = pd.DataFrame({'time': time, 'flux': flux, 'star_radius': star_radius, 'planet_radius': planet_radius})
    
    first_dip = np.where(np.diff(flux) < 0)[0][0]
    first_full = np.where(np.diff(flux) < 0)[0][-1]
    sec_dip = np.where(np.diff(flux) > 0)[0][0]
    sec_full = np.where(np.diff(flux) > 0)[0][-1]

    init_notebook_mode(connected = True)
    fig = go.Figure(
        data=[go.Scatter(x=dfs.time.values, y=dfs.flux.values,
                         mode="lines",
                         name="Imaginary Planet",
                         line=dict(width=2, color="blue")),
              go.Scatter(x=dfs.time.values, y=dfs.flux.values,
                         mode="lines",
                         name="Light from a Star",
                         line=dict(width=2, color="blue"))],
        layout=go.Layout(
            xaxis=dict(range=[0, imsize[0]], autorange=False, zeroline=False),
            yaxis=dict(range=[0, 1.3], autorange=False, zeroline=False),
            title_text="Exoplanet Transit", hovermode="closest",
            xaxis_title='Time',
            yaxis_title='Flux',
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None])])]),
        frames=[go.Frame(
            data=[go.Scatter(
                x=[dfs.time.values[::5][k]],
                y=[dfs.flux.values[::5][k]],
                mode="markers",
                marker=dict(color="red", size=10))])

            for k in range(dfs.time.values[::5].shape[0])]
    )

    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                x=time[first_dip],
                y=flux[first_dip],
                xref="x",
                yref="y",
                text="Planet starts to occult the star",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            ),        
            dict(
                x=time[first_full],
                y=flux[first_full],
                xref="x",
                yref="y",
                text="Planet is fully in front of the star",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            ), 
            dict(
                x=time[sec_dip],
                y=flux[sec_dip],
                xref="x",
                yref="y",
                text="Planet has reached the edge of the star",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=40,

            ),
            dict(
                x=time[sec_full],
                y=flux[sec_full],
                xref="x",
                yref="y",
                text="Planet has stopped occulting the star",
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            )
        ]
    )

    return fig
