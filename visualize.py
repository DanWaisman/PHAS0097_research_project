import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from tqdm.auto import tqdm

# =================================================================================================== #

def pano_plot(x, y, subfigs, patch_shape=(3, 3), ax_0=None):
    '''
    Visualizes a scatter plot with images positioned at each coordinate point.

    This function is designed to create 'panoramic' scatter plots, which are particularly effective for illustrating ...
    ... the distribution of images within feature spaces, aiding in the tasks of clustering and classification.

    INPUTS:
    - x, y => arrays with n elements each, representing the x and y coordinates for the plot, respectively.
    - subfigs => An array containing n images to be placed at the corresponding coordinates. Each image should have...
                ...the dimensions (H, W) for grayscale images or (H, W, C) for color (RGB) images.
    - patch_shape => The dimensions for the thumbnails that will be shown at each coordinate point.
    - ax_0 => Specifies the axis object where the plot will be drawn. If None is passed, the function will create...
             ...a new figure and axis for the plot. Otherwise, the visualization will be added to the provided axis object.
    '''

    if ax_0 is None:
        fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    else:
        ax = ax_0
    
    # Changing patch_shape to same dimensions of plot
    figure_w, figure_h = ax.get_figure().get_size_inches()
    
    figure_dpi = ax.get_figure().get_dpi()
    
    px = patch_shape[0] * figure_w * figure_dpi
    py = patch_shape[1] * figure_h * figure_dpi
    
    # Make sure x and y limits include all points
    ax.set_xlim( min(x) - patch_shape[0] , max(x) + patch_shape[0] )
    ax.set_ylim( min(y) - patch_shape[1] , max(y) + patch_shape[1] )

    for x_j, y_j, img in tqdm(zip(x, y, subfigs)):
        
        # Re-shape subfigs to patch_shape
        img_resized = plt.imshow(img, extent=[x_j - patch_shape[0], x_j + patch_shape[0], y_j - patch_shape[1], y_j + patch_shape[1]], aspect='auto')

    if ax_0 is None:
        plt.show()

# =================================================================================================== #

def pretty_cm(conf_matrix, ordered_labels, colour_scale=0.6, ax_0=None, font_size=6, colour_map='cool'):
    '''
    Creates an enhanced confusion matrix visualization for easy analysis.

    This function arranges the actual labels along the rows and the predicted labels along the columns...
    ... for a straightforward comparison.

    INPUTS:
    - conf_matrix    => An nxn matrix that encapsulates the confusion matrix data.
    - ordered_labels => An ordered list of the class names as they correspond to the rows and columns in the confusion matrix...
                        ... for instance, the first item should match the class of the first row and column in *cm*.
    - colour_scale => A scaling factor for color saturation, enhancing visibility for matrices with few errors and...
                      ...moderating it for those with numerous inaccuracies.
    - ax_0 => Determines the axis object for the plot. Passing None generates a new figure and axis for...
              ... the visualization. Providing an existing axis object will render the matrix directly onto it.
    - font_size  => The font size to be used for the matrix's text elements.
    - colour_map => The color map from matplotlib to apply to the visualization.
    '''
    
    if ax_0 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        fig.set_facecolor('w')
    else:
        ax = ax_0

    n_clusters = len(ordered_labels)
    
    ax.imshow( np.power( conf_matrix, colour_scale ), cmap=colour_map, extent=( 0, n_clusters, 0, n_clusters ) )
    
    tick_label = np.arange( n_clusters ) + 0.5
    
    ax.set_xticks( tick_label, minor=True )
    ax.set_yticks( tick_label, minor=True )
    ax.set_xticklabels( ['' for i in range( n_clusters )], minor=False, fontsize=font_size )
    ax.set_yticklabels( ['' for i in range( n_clusters )], minor=False, fontsize=font_size )
    
    ax.set_xticks(np.arange( n_clusters ))
    ax.set_yticks(np.arange( n_clusters ))
    ax.set_xticklabels( labels = ordered_labels, minor = True, fontsize = font_size )
    ax.set_yticklabels( labels = reversed(ordered_labels), minor = True, fontsize = font_size )

    ax.set_xlabel( 'Predicted Labels', fontsize = font_size )
    ax.set_ylabel( 'Actual Labels',    fontsize = font_size )
    
    for (i, j), z in np.ndenumerate(conf_matrix):
        
        ax.text( j + 0.5, n_clusters - i - 0.5,
                '{:^5}'.format(z), ha='center',
                va='center', fontsize=font_size,
                bbox=dict(boxstyle='round', facecolor='w',edgecolor='0.3') )
    
    ax.grid(which='major', color=np.ones(3) * 0.33, linewidth=1)

    if ax_0 is None:
        
        ax.set_title( 'Accuracy: {:.1f}%'.format( (conf_matrix.trace() / conf_matrix.sum() )*100),
                     fontsize = font_size + 2 )
        plt.show()
        return
    
    else:
        return ax

# =================================================================================================== #

