import numpy as np


def custom_plot(
    self,
    img_order=None,
    freq=None,
    figsize=None,
    no_axis=False,
    mic_marker_size=10,
    plot_directivity=True,
    ax=None,
    **kwargs,
):
    """Plots the room with its walls, microphones, sources and images"""

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = fig.add_subplot(111, aspect="equal", **kwargs)

    # draw room
    for wall in self.walls:
        xs, ys = wall.corners
        absorption = wall.absorption

        if np.array(absorption).mean() != 1:
            color = "brown"
            ax.plot(xs, ys, color=color)

    if self.mic_array is not None:

        for i in range(self.mic_array.nmic):
            ax.scatter(
                self.mic_array.R[0][i],
                self.mic_array.R[1][i],
                marker="x",
                linewidth=0.5,
                s=mic_marker_size,
                c="k",
            )

    # define some markers for different sources and colormap for damping
    markers = ["o", "s", "v", "."]
    cmap = plt.get_cmap("YlGnBu")

    # use this to check some image sources were drawn
    has_drawn_img = False

    # draw the scatter of images
    for i, source in enumerate(self.sources):
        # draw source
        ax.scatter(
            source.position[0],
            source.position[1],
            c=[cmap(1.0)],
            s=20,
            marker=markers[i % len(markers)],
            edgecolor=cmap(1.0),
        )

        # draw images
        if img_order is None:
            img_order = 0
        elif img_order == "max":
            img_order = self.max_order

        I = source.orders <= img_order
        if len(I) > 0:
            has_drawn_img = True

        val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.0) / 10.0
        # plot the images
        ax.scatter(
            source.images[0, I],
            source.images[1, I],
            c=cmap(val),
            s=20,
            marker=markers[i % len(markers)],
            edgecolor=cmap(val),
        )

    # When no image source has been drawn, we need to use the bounding box
    # to set correctly the limits of the plot
    if not has_drawn_img or img_order == 0:
        bbox = self.get_bbox()
        ax.set_xlim(bbox[0, :])
        ax.set_ylim(bbox[1, :])

    return fig, ax
