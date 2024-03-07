import scipy
import pickle
import warnings
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt

######################################################################################
# Functions used for the manuscript "Learning leaves a memory trace in motor cortex" #
# See the Jupyter notebook (mt_example_code), which generates the examples, using    #
# this file as a library.                                                            #
######################################################################################

#############################################################
########## Linear Discriminate Analysis (LDA) ###############
#############################################################
def calc_LDA_Sw_Sb(X, y, equal_priors=True):
    """
    Calculates the within and between class sum of squares for use in LDA.

    Parameters
    ----------
    X : nxd. Number of datapoints by number of features
    y : n class labels

    Returns
    -------
    S_w : The within class sum of squares.
    S_b : The between class sum of squares.
    """
    n, d = X.shape    
    class_set = np.unique(y)
    c = len(class_set)
    
    #  [ [ mean vector of classs 1]
    #    [ mean vector of classs 2] ...]
    m = np.zeros(shape=(c, d) )    
    S = np.zeros(shape=(d, d, c))
    for class_idx, class_val in enumerate(class_set):
        v = X[y == class_val, :] # n x d
        nc = v.shape[0] # number of data points in class c
        if equal_priors: nc = n / c
        mean = np.mean(v, axis=0)
        m[class_idx, :] = mean
        S[:, :, class_idx] = nc * np.cov(  (v - mean).T, bias=True)
        # np.testing.assert_almost_equal((v - mean).T @ (v - mean), nc * np.cov(  (v - mean).T, bias=True), decimal=5 )
        S[:, :, class_idx] = (v - mean).T @ (v - mean)
    S_w = np.sum(S, axis=-1) # dxd
    assert S_w.shape == (d, d)
    S_w = (S_w + S_w.T) / 2 # numerical stablitiy. S_w is already symmetric.
    # m is class x features. mi is m[i, :].
    # S_b = n * np.cov(m.T, bias=True) # dxd. N in covar formula is c. (divide by c cancles with multiply by c)  
    overall_mean = np.mean(X, axis=0).reshape((d, 1))
    assert overall_mean.shape == (d, 1)
    S_b = np.zeros((d,d))
    for class_idx, class_val in enumerate(class_set):
        v = X[y == class_val, :] # n x d
        nc = v.shape[0] # number of data points in class c
        if equal_priors: nc = n / c
        mean = m[class_idx, :].reshape((d, 1))
        S_b += nc * (mean - overall_mean) @ (mean - overall_mean).T    
    S_b = (S_b + S_b.T) / 2 # numerical stablitiy. S_b is already symmetric.
    assert S_b.shape == (d, d)
    return S_w, S_b # both are dxd matrices.

def calc_lda(X, y, k=None):
    """    
    Parameters
    ----------
    X : nxd. Number of datapoints by number of features
    y : n class labels
    k : number of dimensions to reduce to. Defaults to c - 1, where c is the
        number of classes

    Returns
    -------
    eig_vals: eigenvalues.
    eig_vecs: The eigenvectors. This is W. This is dxk.
              Z = X*W.
    """
    if k is None: k = len(set(y)) - 1
    S_w, S_b = calc_LDA_Sw_Sb(X, y)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
    return eig_vals[:k], np.real_if_close(eig_vecs[:, :k])


#####################################################
###### Hypothesis testing / Permutation tests #######
#######################################v#############
def permute_subset(x, to_permute_subset, copy_x=True):
    """ Takes array x and a list of elements to permute in that array. Permutes only those elements,
    leaving the rest alone. If copy_x we won't modify x in place.
    Example:
        x = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        to_permute_subset = [1, 3]
        putil_array.permute_subset(x, to_permute_subset, copy_x=True) ->   array([0, 1, 2, 1, 4, 0, 3, 2, 3, 4])
        
        
        x = [0, 'a', 'b', 4, 'a', 2, 'b', 4]
        to_permute_subset = ['a', 'b']
        h.permute_subset(x, to_permute_subset, copy_x=True)
        array(['0', 'b', 'a', '4', 'b', '2', 'a', '4'], dtype='<U21')
    """
    x = np.squeeze(np.asarray(x))
    assert x.ndim == 1
    if copy_x: x = np.copy(x)
    permute_flag = np.isin(x, to_permute_subset)
    per_array = x[permute_flag]
    x[permute_flag] = np.random.permutation(per_array)
    return x

def univariate_ftest(x, y, p2=2):
    """
    Does a univariate_ftest on x and y. Tested to give the same results as Matlab and sklearn.
    

    Examples to compare againts:
    
    Matlab code:
        x = [3504;3693;3436;3433;3449;4341;4354;4312;4425;3850;3090;4142;4034;4166;3850;3563;3609;3353];
        y = [18;15;18;16;17;15;14;14;14;15;12;13;15;10;12;15;14;10];
        tbl = table(y,x);        
        mdl = fitlm(tbl,'y ~ x')

    Python code:
        x = np.asarray([3504,3693,3436,3433,3449,4341,4354,4312,4425,3850,3090,4142,4034,4166,3850,3563,3609,3353])
        y = np.asarray([18,15,18,16,17,15,14,14,14,15,12,13,15,10,12,15,14,10])
        print(sklearn.feature_selection.f_regression(x.reshape(-1, 1), y))

    Parameters
    ----------
    x : univariate data
    y : response variable
    p2 : number of parameters. Should always be 2 (y = mx + b). m and b are the
         two parameters.

    Returns
    -------
    F : F statistic
    pval : pvalue
    """
    assert len(x) == len(y)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    slope, intercept, r, p, se = scipy.stats.linregress(x, y)
    y2_pred = x * slope + intercept # unconstrained hypothesis
    y1_pred = np.mean(y) # constrained hypothesis
    n = len(y)
    rss2 = np.sum( (y - y2_pred) ** 2)
    rss1 = np.sum( (y - y1_pred) ** 2)
    p1 = 1 # number of parameters for our constrained hypothesis
    F = ((rss1 - rss2) / (p2 - p1)) / (rss2 / (n - p2 ))
    pval = scipy.stats.f.sf(F, dfn=1, dfd=n - p2)
    return F, pval


#######################################
###### Angle Functions    #############
#######################################
def calc_angle_between180(vects1, vects2, warn_on_zero_vect=True):
    """
    :param vects1: shape = n x 2
    :param vects2: shape = n x 2

    if vect2 is clockwise of vect1, then the result is positive
    if vect2 is counterclockwise of vect1, then the result is negative
    vect1 can be thought of as the "ground".
    """
    vects1 = convert_to_2Dvector(vects1)
    vects2 = convert_to_2Dvector(vects2)
    if vects1.shape != vects2.shape: raise ValueError()
    angles1 = vects_to_angles360(vects1, warn_on_zero_vect=warn_on_zero_vect)
    angles2 = vects_to_angles360(vects2, warn_on_zero_vect=warn_on_zero_vect)
    angular_diff180 = convert_arbitrary_to_theta180(angles2 - angles1)
    angular_diff180 = round_theta180(angular_diff180)
    return convert_singleton_to_float(angular_diff180)

def vects_to_angles180(vects, warn_on_zero_vect=True):
    """
    Converts a set of vectors to angles in degrees within the range [-180, 180].
    
    Parameters:
    - vects (np.ndarray): An array of shape Nx2 representing vectors, where each row is a vector with x and y components.
    - warn_on_zero_vect (bool, optional): If True, issue a warning when a zero vector is encountered. Defaults to True.
    
    Returns:
    - np.ndarray: An array of angles in degrees corresponding to the input vectors, adjusted to lie within the [-180, 180] range.
    
    Notes:
    - Zero vectors (where both x and y components are 0) will result in NaN angles.
    - The function ensures that the angles are rounded and validated to be within the specified range.
    """
    vects = convert_to_2Dvector(vects)
    xcoords = vects[:, 0]
    ycoords = vects[:, 1]
    is_zero_vect = np.logical_and(xcoords == 0, ycoords == 0)
    if warn_on_zero_vect and np.any(is_zero_vect): warnings.warn('Zero Vector Encountered')
    xcoords = np.where(is_zero_vect, np.nan,    xcoords)
    ycoords = np.where(is_zero_vect, np.nan, ycoords)
    thetas180 = np.rad2deg(np.arctan2(ycoords, xcoords))  # y is supposed to be the first argument.
    thetas180 = round_theta180(thetas180)
    assert_range_theta180(thetas180)
    return convert_singleton_to_float(thetas180)

def convert_arbitrary_to_theta180(arb_theta):
    """
    Converts an arbitrary angle to an angle in degrees within the range [-180, 180].
    
    Parameters:
    - arb_theta: The arbitrary angle(s) to be converted. Can be a scalar, list, or numpy array of any shape.
    
    Returns:
    - The converted angle(s) in degrees within the range [-180, 180].
    """

    return convert_theta360_to_theta180(convert_arbitrary_to_theta360(arb_theta))

def assert_in_radian_range(thetas_rad): 
    """
    Asserts that all provided angles in radians are within the range [0, 2 * pi).
    
    Parameters:
    - thetas_rad (np.ndarray): An array of angles in radians.
    
    Raises:
    - AssertionError: If any angle is not in the specified range.
    """
    assert np.all((0 <= thetas_rad) & (thetas_rad < 2 * np.pi)), thetas_rad

def convert_arbitrary_to_theta360(arb_theta):
    """
    Converts an arbitrary angle to an angle in degrees within the range [0, 360].
    
    Parameters:
    - arb_theta: The arbitrary angle(s) to be converted, can be in any units or range.
    
    Returns:
    - The converted angle(s) in degrees within the range [0, 360].
    """
    arb_theta = convert_to_1Dvector(arb_theta)
    theta360 = arb_theta % 360  # Can handle angles in (-inf to inf)
    assert_range_theta360(theta360)
    return convert_singleton_to_float(theta360)

def round_theta180(theta180):
    """
    Rounds angles very close to -180 degrees to +180 degrees.
    
    Parameters:
    - theta180 (np.ndarray): An array of angles in degrees.
    
    Returns:
    - np.ndarray: An array of angles, where angles within rounding distance of -180 are changed to +180.
    """
    with warnings.catch_warnings(record=True) as w: # catch warnings for nan
        bools = (theta180 < -179.999999)
        if len(w) == 1: assert not np.all(np.isfinite(theta180))
    if np.any(bools):
        theta180 = np.where(bools, 180, theta180)  # numpy.where(condition, be this, else this).
    return theta180

def convert_to_1Dvector(data):
    """
    Converts input data to a 1D numpy array.
    
    Parameters:
    - data: Input data in the form of a list, range, numpy array, or number.
    
    Returns:
    - np.ndarray: Data as a 1D numpy array.
    
    Raises:
    - TypeError: If the input data type is not supported.
    """
    if type(data) is np.ndarray: pass
    elif type(data) is list: data = np.asarray(data)
    elif type(data) is range: data = np.asarray(data)
    elif isinstance(data, (int, float)): data = np.asarray([data])
    else: raise TypeError(f'Invalid Type: {type(data)}')
    assert data.ndim == 1, data.ndim
    return data


def _remove_nan(thetas):
    """
    Removes NaN values from an array of angles.
    
    Parameters:
    - thetas (np.ndarray): An array of angles, which may include NaN values.
    
    Returns:
    - np.ndarray: An array of the same angles with NaN values removed.
    """
    return thetas[np.isfinite(thetas)]

def assert_range_theta360(theta360, ignore_nan=True):
    """
    Asserts that all angles in a vector are within the exclusive range [0, 360) degrees.
    
    Parameters:
    - theta360 (np.ndarray or similar): A vector of angles in degrees.
    - ignore_nan (bool, optional): If True, NaN values are ignored in the assertion. Defaults to True.
    
    Raises:
    - AssertionError: If any angle is exactly 360 degrees or not within the range [0, 360).
    """
    theta360 = convert_to_1Dvector(theta360)
    if ignore_nan: theta360 = _remove_nan(theta360)
    assert np.all(theta360 != 360), np.nanmax(theta360)  # We don't include 360. This should be zero
    assert np.all((0 <= theta360) & (theta360 < 360))

def convert_theta360_to_theta180(theta360):
    """
    Converts angles in degrees from the range [0, 360) to the range [-180, 180).
    
    Parameters:
    - theta360 (np.ndarray or similar): A vector of angles in degrees within the range [0, 360).
    
    Returns:
    - np.ndarray: The converted angles in degrees within the range [-180, 180).
    """
    assert_range_theta360(theta360)
    theta360 = convert_to_1Dvector(theta360)
    with warnings.catch_warnings(record=True) as w:
        theta180 = np.where(theta360 > 180, theta360 % -180, theta360)
        if len(w) == 1: assert not np.all(np.isfinite(theta360))
    theta180 = round_theta180(theta180)
    theta180 = convert_singleton_to_float(theta180)
    assert_range_theta180(theta180)
    return theta180

def assert_range_theta180(theta180, ignore_nan=True):
    """
    Asserts that all angles in a vector are within the range (-180, 180] degrees.
    
    Parameters:
    - theta180 (np.ndarray or similar): A vector of angles in degrees.
    - ignore_nan (bool, optional): If True, NaN values are ignored in the assertion. Defaults to True.
    
    Raises:
    - AssertionError: If any angle is not within the range (-180, 180].
    """
    theta180 = convert_to_1Dvector(theta180)
    if ignore_nan: theta180 = _remove_nan(theta180)
    # If we are in rounding distance to -180, we convert this to +180
    assert np.all((-180 < theta180) & (theta180 <= 180)), f'Ignore Nan?: {ignore_nan}, Min={np.min(theta180)}, Min={np.max(theta180)}'

def calc_rotation_matrix(theta_any):
    """
    Calculates a 2D rotation matrix for rotating vectors by a specified angle.
    
    Parameters:
    - theta_any (float): The rotation angle in degrees. Can be negative for clockwise rotation.
    
    Returns:
    - np.ndarray: The 2x2 rotation matrix. Use this matrix to rotate vectors by left-multiplying: v' = Rv.
    
    Notes:
    - `v` is a column vector in the format [x; y].
    """
    theta = np.radians(theta_any)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array(((c, -s),
                  (s, c)))
    return R

def rotate(origin, point, angle, flipy):
    """
    Rotates a point around a given origin by a specified angle, optionally flipping the y-coordinate.
    
    Parameters:
    - origin (tuple): The (x, y) coordinates of the origin point around which to rotate.
    - point (np.ndarray): The (x, y) coordinates of the point(s) to rotate.
    - angle (float): The rotation angle in degrees.
    - flipy (bool): If True, the y-coordinate of the rotated point is flipped.
    
    Returns:
    - tuple: The (x, y) coordinates of the rotated point(s).
    """
    angle = np.deg2rad(angle)
    ox, oy = origin
    px, py = point[:, 0], point[:, 1]
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if flipy:
        qy = -qy
    return qx, qy


def convert_theta180_to_theta360(theta180):
    """
    Converts angles from the range [-180, 180) to [0, 360), handling angles outside the [-180, 180) range.
    
    Parameters:
    - theta180 (np.ndarray or similar): Angles in degrees to be converted, can include NaNs.
    
    Returns:
    - The converted angles in degrees within the range [0, 360).
    
    Notes:
    - Ensures that the input angles are within the valid [-180, 180) range before conversion.
    """
    theta180 = convert_to_1Dvector(theta180)
    assert_range_theta180(theta180)
    theta360 = theta180 % 360  # Can handle angles in (-inf to inf)
    theta360 = round_theta360(theta360)
    assert_range_theta360(theta360)
    return convert_singleton_to_float(theta360)

def round_theta360(theta360):
    """
    Adjusts angles close to 360 degrees down to 0 degrees, effectively rounding them for continuity.
    
    Parameters:
    - theta360 (np.ndarray): Angles in degrees potentially close to 360.
    
    Returns:
    - np.ndarray: Angles with those within rounding distance of 360 degrees adjusted to 0 degrees.
    """
    with warnings.catch_warnings(record=True) as w: # catch warnings for nan
        bools = (theta360 > 359.9999)
        if len(w) == 1: assert not np.all(np.isfinite(theta360))
    if np.any(bools):
        theta360 = np.where(bools, 0, theta360) # numpy.where(condition, be this, else this).
    return theta360

def vects_to_angles360(vects, warn_on_zero_vect=True):
    """
    Converts a set of vectors to angles in degrees within the range [0, 360).
    
    Parameters:
    - vects (np.ndarray): Input vectors in Nx2 format.
    - warn_on_zero_vect (bool, optional): If True, issues a warning for zero vectors. Defaults to True.
    
    Returns:
    - np.ndarray: Angles corresponding to the input vectors, adjusted to lie within the [0, 360) range.
    """
    return convert_theta180_to_theta360(vects_to_angles180(vects, warn_on_zero_vect))

def angles_to_unit_vectors(thetaAny, round_small_epsilon=False):
    """
    Converts angles in degrees to unit vectors.
    
    Parameters:
    - thetaAny (np.ndarray or similar): Angles in degrees, can be in any range (theta360, theta180, etc.).
    - round_small_epsilon (bool or float, optional): If True or specified as a float, small vector components are rounded to zero. Defaults to False.
    
    Returns:
    - np.ndarray: Unit vectors corresponding to the input angles. Shape is N x 2 or (2,) for single angle input.
    """
    thetaAny = convert_to_1Dvector(thetaAny)
    x = np.cos(np.deg2rad(thetaAny))
    y = np.sin(np.deg2rad(thetaAny))
    unit_vect = np.asarray([x, y]).T
    assert unit_vect.ndim == 2
    assert unit_vect.shape[1] == 2
    if round_small_epsilon: unit_vect = _round_small_to_zero(unit_vect, epsilon=round_small_epsilon)
    return unit_vect.squeeze()

def _round_small_to_zero(a, epsilon=10**-12):
    """
    Rounds small elements of an array to zero.
    
    Parameters:
    - a (np.ndarray): Input array of any shape.
    - epsilon (float, optional): Threshold below which values are rounded to zero. Defaults to 10**-12.
    
    Returns:
    - np.ndarray: The modified array with elements less than epsilon set to zero.
    """
    if type(epsilon) is bool: epsilon=10**-12
    a[np.abs(a) < epsilon] = 0
    return a

def convert_singleton_to_float(data):
    """
    Converts input data to a float, if applicable.
    
    Parameters:
    - data: Input data, can be a numpy array, int, or float.
    
    Returns:
    - Float or unmodified numpy array, depending on the shape and type of input.
    
    Raises:
    - ValueError: If the input data type is not supported.
    """
    if type(data) is np.ndarray:
        if data.shape == (1,): return float(data)
        else: return data
    elif isinstance(data, (int, float)): return float(data)
    else: raise ValueError(data)

def convert_to_2Dvector(data):
    """
    Converts input data to a 2D vector format, specifically for data with two components per vector.
    
    Parameters:
    - data (np.ndarray): Input data, can be a single vector of shape (2,) or multiple vectors of shape (n, 2).
    
    Returns:
    - np.ndarray: The input data formatted as an array of 2D vectors. If the input is a single vector, it is expanded to shape (1, 2).
    
    Raises:
    - AssertionError: If the input data is not in the correct shape or if `p` is not equal to 2.
    - NotImplementedError: If the input data has a dimensionality of 0.
    - TypeError: If the input data is not a numpy array.
    """
    assert type(data) is np.ndarray, print(type(data))
    if ((data.ndim == 2) and (data.shape[1] != 2)) :
        raise AssertionError(f'Passed data should have been shape nx2. Data was {data.shape}')

    if data.ndim == 0: raise NotImplementedError()
    if data.ndim == 1: data = np.expand_dims(data, axis=0)
    if (data.shape[1] != 2 or data.ndim != 2):
        raise AssertionError(f'Data should have been shape nx2. Data was {data.shape}')
    return data

################################
###### Covariance  #############
################################
def eigsorted(cov):
    """
    Computes the eigenvalues and eigenvectors of a covariance matrix, sorted by eigenvalue in descending order.
    
    Parameters:
    - cov (np.ndarray): The covariance matrix to decompose.
    
    Returns:
    - tuple: A pair (vals, vecs) where `vals` are the sorted eigenvalues and `vecs` are the corresponding eigenvectors.
    """
    vals, vecs = np.linalg.eigh(cov)
    order_ = vals.argsort()[::-1]
    return vals[order_], vecs[:, order_]

def plot_covar_ellipse(ax, x, y, nstd=1, color='k', invis_scatter=True, show_mean=True,
                       linewidth=None, show_mean_size=30, label='__no_legend__'):
    """
    Plots an ellipse representing the covariance of a dataset, with optional features.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the ellipse.
    - x, y (np.ndarray): Data points to calculate covariance for.
    - nstd (int, optional): Number of standard deviations to determine the ellipse's radii. Defaults to 1.
    - color (str, optional): Color of the ellipse. Defaults to 'k' (black).
    - invis_scatter (bool, optional): If True, scatter points are plotted invisibly. Defaults to True.
    - show_mean (bool, optional): If True, plot the mean of the data. Defaults to True.
    - linewidth (float, optional): Line width of the ellipse. Defaults to None (uses default line width).
    - show_mean_size (int, optional): Size of the mean point marker. Defaults to 30.
    - label (str, optional): Label for the mean point marker. Defaults to '__no_legend__'.
    
    Notes:
    - The function adds the ellipse as a non-filled, color-specified matplotlib.patches.Ellipse object to the axes.
    """
    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = matplotlib.patches.Ellipse(xy=(np.nanmean(x), np.nanmean(y)), width=w, height=h, angle=theta, color=color, linewidth=linewidth, zorder=700)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    if invis_scatter: ax.scatter(x, y, alpha=0.0)
    if show_mean:
        ax.scatter(np.mean(x), np.mean(y), s=show_mean_size, color=color, label='__no_legend__', zorder=200)

def calc_mahalanobis(cov, diff):
    """
    Calculates the Mahalanobis distance between a point and a distribution.
    
    Parameters:
    - cov (np.ndarray): The covariance matrix of the distribution.
    - diff (np.ndarray): The difference vector between the point and the mean of the distribution.
    
    Returns:
    - float: The Mahalanobis distance.
    
    Notes:
    - The covariance matrix must be square and match the dimensionality of the difference vector.
    - This distance is a measure of the distance between the point and the distribution, taking into account the covariance of the distribution.
    """
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(diff.T @ inv_cov @ diff)

#######################################
###### Load or Save Data  #############
#######################################
def load_pickle(file_name, verbose=True):
    """
    Loads a pickle file into a Python object.
    
    Parameters:
    - file_name (str): Name of the pickle file to load, with or without the '.pickle' extension.
    - verbose (bool, optional): If True, prints a message upon successful loading. Defaults to True.
    
    Returns:
    - The Python object loaded from the pickle file.
    """
    file_name = file_name.replace('.pickle', '')
    with open(file_name + '.pickle', 'rb') as handle:
        x = pickle.load(handle)
    return x


###############################################
########### Fonts, Colors, Plotting ###########
###############################################
def _to_ax_iterable(axes):
    """
    Converts the input into an iterable form of matplotlib.axes.Axes.
    
    Parameters:
    - axes: A single matplotlib.axes.Axes object, an iterable of Axes objects, or None.
    
    Returns:
    - An iterable (list) of matplotlib.axes.Axes objects. If `axes` is None, returns the current axes in a list.
    """
    if axes is None: axes = plt.gca()
    if _is_iterable(axes): return axes
    else: return [axes]
    

def _is_iterable(arr):
    """
    Determines if the passed object is iterable.
    
    Parameters:
    - arr: The object to check for iterability.
    
    Returns:
    - bool: True if the object is iterable, False otherwise.
    """
    try:
        iter(arr)
        return True
    except TypeError: return False


def task_to_color(task):
    """
    Converts a task identifier to a hexadecimal color value.
    
    Parameters:
    - task (str): The identifier of the task ('FamTask1', 'NewTask', 'FamTask2').
    
    Returns:
    - str: The hexadecimal color value associated with the given task.
    
    Raises:
    - ValueError: If `task` is not one of the predefined identifiers.
    """
    if   task == 'FamTask1': return '#ff6666'  # watermellon
    elif task == 'NewTask' : return '#000080'  # navy blue
    elif task == 'FamTask2' : return '#700000' # dark red
    else: raise ValueError(task)

def set_standard_fonts(ax=None, ticksize=5, labelsize=6.5):
    """
    Sets standard font sizes for axes labels and ticks.
    
    Parameters:
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify. If None, applies to current axes.
    - ticksize (int or float, optional): Font size for tick labels. Defaults to 5.
    - labelsize (int or float, optional): Font size for axes labels. Defaults to 6.5.
    """
    axes = _to_ax_iterable(ax)
    for ax in axes:

        ax.xaxis.get_label().set_fontsize(labelsize)
        ax.yaxis.get_label().set_fontsize(labelsize)
        for tick in ax.get_xticklabels(): tick.set_fontsize(ticksize)
        for tick in ax.get_yticklabels(): tick.set_fontsize(ticksize)

def hide_spine(tblr='tr', ax=None):
    """
    Hides specified spines of an axes object.
    
    Parameters:
    - tblr (str): String containing characters 't', 'b', 'l', and/or 'r' to specify top, bottom, left, and right spines, respectively.
    - ax (matplotlib.axes.Axes, optional): Axes object whose spines to hide. If None, applies to current axes.
    
    Raises:
    - ValueError: If `tblr` contains invalid characters.
    """
    for top_bttm_left_right in tblr:
        if   top_bttm_left_right.startswith('l'): ax.spines['left'].set_visible(False)
        elif top_bttm_left_right.startswith('r'): ax.spines['right'].set_visible(False)
        elif top_bttm_left_right.startswith('t'): ax.spines['top'].set_visible(False)
        elif top_bttm_left_right.startswith('b'): ax.spines['bottom'].set_visible(False)
        else: raise ValueError(tblr)

def block_type_to_color(block_type): 
    """
    Converts a block type string to a corresponding hex color value.
    
    Parameters:
    - block_type (str): The type of block to convert to color.
    
    Returns:
    - str: Hex color code corresponding to the block type.
    
    Raises:
    - ValueError: If `block_type` is not a recognized type.
    """
    if   block_type == 'FamTask1': return '#ff6666' # watermellon
    elif block_type == 'NewTask' : return '#000080' # navy blue
    elif block_type == 'FamTask2': return '#700000' # dark red
    else: raise ValueError(block_type)
    
def set_tick_locations(locations, xy, ax=None):
    """
    Sets the locations of ticks on the x or y axis.
    
    Parameters:
    - locations (list of int or float): The locations to place the ticks.
    - xy (str): 'x' to set x-axis ticks, 'y' to set y-axis ticks, or 'xy' to set both.
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify. If None, applies to current axes.
    """

    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if 'x' in xy: ax.set_xticks(locations)
        if 'y' in xy: ax.set_yticks(locations)

def calc_zero_centered_hist_bins(y, bin_width):
    """
    Calculates histogram bins centered around zero with a specified bin width.
    
    Parameters:
    - y (np.ndarray): Data array for which to calculate bins.
    - bin_width (float): Width of each bin.
    
    Returns:
    - list: Bin edges for the histogram.
    """
    bins = list(np.arange( (np.floor(np.min(y)) // bin_width) * bin_width , 0, bin_width)) + list(np.arange(0, (np.ceil(np.max(y)) // bin_width) * bin_width + 2 * bin_width, bin_width))
    assert np.min(y) > bins[0]
    assert np.max(y) < bins[-1], (np.max(y), bins[-1])
    return bins


def hide_tick_marks_and_labels(xy='xy', ax=None):
    """
    Hides tick marks and labels on the specified axes.
    
    Parameters:
    - xy (str): Specifies which axis('axes') to affect: 'x', 'y', 'xy' (both), 'z' (3D plots), or combinations for 3D.
    - ax (matplotlib.axes.Axes, optional): Axes object to modify. If None, modifies the current axes.
    """
    if ax is None: ax = plt.gca()
    if xy == 'xy':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif xy == 'x': ax.get_xaxis().set_ticks([])
    elif xy == 'y': ax.get_yaxis().set_ticks([])
    elif xy == 'z': ax.get_zaxis().set_ticks([])
    elif xy == 'xy' or xy == 'yx':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif xy == 'xyz':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])
    else: raise ValueError(xy)

def hide_tick_marks(xy, ax=None):    
    """
    Hides tick marks on specified axes without affecting the labels.
    
    Parameters:
    - xy (str): 'x' for x-axis, 'y' for y-axis, 'xy' or 'both' for both axes.
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify. If None, modifies the current axes.
    """
    if ax is None: ax = plt.gca()
    axes = _to_ax_iterable(ax)

    for _, ax in enumerate(axes):
        if   xy == 'x': ax.tick_params(axis='x', which='both',length=0)
        elif xy == 'y': ax.tick_params(axis='y', which='both',length=0)
        elif (xy == 'xy') or (xy == 'both'): ax.tick_params(axis='both', which='both',length=0)


def show_tick_labels(xy='xy', ax=None):
    """
    Enables tick labels on specified axes.
    
    Parameters:
    - xy (str): Specifies which axis('axes') to enable labels for: 'x', 'y', or 'xy'.
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify.
    """
    assert 'x' in xy or 'y' in xy, xy
    axes = _to_ax_iterable(ax)
    for ax in axes:
        if 'x' in xy: ax.xaxis.set_tick_params(labelbottom=True)
        if 'y' in xy: ax.yaxis.set_tick_params(labelbottom=True)

def set_tick_label_fontsize(both=None, x_tick_label_fontsize=None, y_tick_label_fontsize=None, ax=None):
    """
    Sets the font size of tick labels on specified axes.
    
    Parameters:
    - both (int, optional): Font size to apply to both x and y tick labels. If specified, overrides individual sizes.
    - x_tick_label_fontsize (int, optional): Font size for x-axis tick labels.
    - y_tick_label_fontsize (int, optional): Font size for y-axis tick labels.
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify.
    """
    axes = _to_ax_iterable(ax)
    if both is not None:
        if type(both) is bool:
            raise ValueError("Both is supposed to be an int, not bool")
        x_tick_label_fontsize = both
        y_tick_label_fontsize = both
    try:   len(axes)
    except TypeError: raise
    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            if x_tick_label_fontsize is not None: tick.label1.set_fontsize(x_tick_label_fontsize)
        for tick in ax.yaxis.get_major_ticks():
            if y_tick_label_fontsize is not None: tick.label1.set_fontsize(y_tick_label_fontsize)

def plot_bar_with_stars(mean, top_data_lim, pval, left_dist_mean=0, ax=None, color=None, star_fontsize=10, linewidth=2):
    """
    Plots a bar with significance stars based on p-value at a specified location.
    
    Parameters:
    - mean (float): The mean value for positioning the stars.
    - top_data_lim (float): The upper limit for positioning the stars.
    - pval (float): The p-value to determine the number of stars.
    - left_dist_mean (float, optional): Leftmost position for the bar. Defaults to 0.
    - ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, uses current axes.
    - color (str, optional): Color of the bar and stars. Defaults to 'k' (black).
    - star_fontsize (int, optional): Font size of the stars. Defaults to 10.
    - linewidth (int, optional): Width of the line. Defaults to 2.
    """
    if ax is None: ax = plt.gca()
    if color is None: color = 'k'
    if pval > 0.05: star_fontsize = 5
    stars = pval_to_stars(pval)
    add_text( (mean + left_dist_mean) / 2, top_data_lim, stars, ax=ax, fontsize=star_fontsize, coord_type='data', va='bottom', 
             ha='center', fontname='DejaVu Sans', color=color)
    ax.plot( [left_dist_mean, mean], [top_data_lim, top_data_lim], color=color, linewidth=linewidth, solid_capstyle='butt')

def pval_to_stars(pval):
    """
    Converts a p-value to a string of significance stars.
    
    Parameters:
    - pval (float): The p-value to convert.
    
    Returns:
    - str: A string of stars representing the significance level, or 'n.s.' for not significant.
    """

    if pval < 0.001: stars = '***'
    elif pval < 0.01: stars = '**'
    elif pval < 0.05: stars = '*'
    else: stars = 'n.s.'
    return stars

def add_text(x, y, text, ha='auto', va='auto', ax=None, fontsize=10, coord_type='axis', fontname='DejaVu Sans', **kwargs):
    """
    Adds text to the plot at specified coordinates with various formatting options.
    
    Parameters:
    - x, y (float): The coordinates for the text.
    - text (str): The text to add.
    - ha (str, optional): Horizontal alignment ('left', 'center', 'right'). Defaults to 'auto'.
    - va (str, optional): Vertical alignment ('top', 'bottom', 'center'). Defaults to 'auto'.
    - ax (matplotlib.axes.Axes, optional): Axes object to add text to. If None, uses current axes.
    - fontsize (int, optional): Font size of the text. Defaults to 10.
    - coord_type (str, optional): The coordinate system ('axis' or 'data'). Defaults to 'axis'.
    - fontname (str, optional): The font name. Defaults to 'DejaVu Sans'.
    - **kwargs: Additional keyword arguments for `ax.text`.
    """
    if ax is None: ax = plt.gca()
    if ha == 'auto':
        if x < 0.5: ha = 'left'
        elif x > 0.5: ha = 'right'
        elif x == 0.5: ha = 'center'
    if va == 'auto':
        if y < 0.5: va = 'bottom'
        elif y > 0.5: va = 'top'
        elif y == 0.5: va = 'center'

    if coord_type in {'axis', 'ax'}:
            ax.text(x, y, text, verticalalignment=va, horizontalalignment=ha, fontsize=fontsize, transform=ax.transAxes, fontname=fontname, **kwargs)
    elif coord_type == 'data':
            ax.text(x, y, text,
        verticalalignment=va, horizontalalignment=ha, fontsize=fontsize, fontname=fontname, **kwargs)
    else: raise ValueError(coord_type)

def map_to_color(map_type):
    """
    Converts a map type to a color.
    
    Parameters:
    - map_type (str): The type of map ('FamMap' or 'NewMap').
    
    Returns:
    - str: The hex color code associated with the map type.
    
    Raises:
    - ValueError: If `map_type` is unrecognized.
    """
    if   map_type == 'FamMap': return '#cc000fff' # red
    elif map_type == 'NewMap': return '#0000ccff' # blue
    else: raise ValueError()

def set_tick_labels(labels, xy, ax=None):
    """
    Sets custom tick labels on the specified axis('axes').
    
    Parameters:
    - labels (list of str): The labels to set.
    - xy (str): 'x' for x-axis, 'y' for y-axis.
    - ax (matplotlib.axes.Axes or iterable of Axes, optional): Axes object(s) to modify.
    """
    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if 'x' in xy: ax.set_xticklabels(labels)
        if 'y' in xy: ax.set_yticklabels(labels)
        if not (('x' in xy) or ('y' in xy)):
            raise ValueError('x or y should be passed as xy')


############################
####### Figure 4 ###########
############################
def fig4_rotate_data(df, columns, rotation_matrix):
    """
    Transforms data using a rotation matrix.
    """
    transformed_data = df[columns].values @ rotation_matrix
    return transformed_data[:, 0], transformed_data[:, 1]

def fig4_plot_trajectory(ax, x, y, color_map, online_map):
    """
    Plots trajectory and scatter points on the axis.
    """
    ax.plot(x, y, color=color_map(online_map), zorder=500, linewidth=0.5)
    ax.scatter(x, y, color=color_map(online_map), zorder=501, s=1)

def fig4_plot_velocity_vectors(ax, x, y, vel_x, vel_y, color_map, offline_map, scale):
    """
    Plots velocity vectors on the axis.
    """
    for i in range(4, len(x)):
        ax.arrow(x[i], y[i], vel_x[i] * scale, vel_y[i] * scale, length_includes_head=True, color=color_map(offline_map), width=0.9, head_width=2.7, head_length=3.5, zorder=5000, ec='none')

def fig4_set_plot_limits(ax, plot_NewTasks):
    """
    Sets the limits of the plot based on given flags.
    """
    if plot_NewTasks:
        ax.set_ylim([-30, 30])
        ax.set_xlim([-20, 150])
    else:
        ax.set_ylim([-25, 25])
        ax.set_xlim([-20, 150])


def fig4b_check_bins(bins, famtask1_prog, famtask2_prog, plot_newtask, newtask_prog=None):
    """
    Check if the program data for familiar tasks and new task (if plotted) fall within the specified bins.

    Args:
    bins (array): The bins used for histogram plotting.
    famtask1_prog (array): Progress data for the first familiar task.
    famtask2_prog (array): Progress data for the second familiar task.
    plot_newtask (bool): Flag to indicate whether to plot the new task.
    newtask_prog (array, optional): Progress data for the new task. Defaults to None.

    Raises:
    AssertionError: If any of the progress data falls outside the specified bins.
    """
    # Ensuring that the minimum progress values of the familiar tasks are greater than the minimum bin value
    assert np.min(famtask1_prog) > np.min(bins), (np.min(famtask1_prog), np.min(bins))
    assert np.min(famtask2_prog) > np.min(bins), (np.min(famtask2_prog), np.min(bins))
    # Ensuring that the maximum progress values of the familiar tasks are less than the maximum bin value
    assert np.max(famtask1_prog) < np.max(bins), (np.max(famtask1_prog), np.max(bins))
    assert np.max(famtask2_prog) < np.max(bins), (np.max(famtask2_prog), np.max(bins))
    # If plotting new task, check its progress data against the bins as well
    if plot_newtask:
        assert np.max(newtask_prog) < np.max(bins), (np.max(newtask_prog), np.max(bins))
        assert np.max(newtask_prog) < np.max(bins), (np.max(newtask_prog), np.max(bins))


def fig4b_plot_example_trials(ax, famtask1_example_trial, famtask2_example_trial):    
    """
    Plot example trials for familiar tasks on a given axis.

    Args:
    famtask1_example_trial (array): Example trial data for the first familiar task.
    famtask2_example_trial (array): Example trial data for the second familiar task.
    """
    # Plotting the mean of the example trials for each familiar task
    ax.scatter([np.mean(famtask1_example_trial)], [0.25], color=task_to_color('FamTask1'), s=1)
    ax.scatter([np.mean(famtask2_example_trial)], [0.25], color=task_to_color('FamTask2'), s=1)


def fig4b_set_plot_formatting(ax, map_type):
    """
    Set the formatting for the plot based on the given axis and map type.

    Args:
    ax (matplotlib.axes.Axes): The axis to format.
    map_type (str): Type of the map, used to determine color and label settings.
    """
    # Applying standard formatting and labels to the axis
    show_tick_labels(ax=ax)
    set_standard_fonts(ax)
    hide_spine(tblr='tr', ax=ax)  # Hiding the top and right spines
    ax.spines['bottom'].set_edgecolor(map_to_color(map_type))  # Coloring the bottom spine
    ax.tick_params(axis='x', colors=map_to_color(map_type))  # Coloring the x-axis ticks
    ax.set_ylabel('Number of Trials')
    ax.set_xlim([-6, None])
    ax.set_ylim([0, 15])
    set_tick_locations([0, 5, 10, 15], 'y')  # Setting tick locations on y-axis

    # Setting the x-axis label based on the map type
    if map_type == 'FamMap':
        ax.set_xlabel('Progress through\nonline Familiar Map (mm/s)', color=map_to_color(map_type))
    else:
        ax.set_xlabel('Progress through\noffline New Map (mm/s)', color=map_to_color(map_type))

def fig4b_get_bins(plot_newtask):
    """
    Generate bins for histogram plotting based on whether the new task is being plotted.

    Args:
    plot_newtask (bool): Flag to indicate whether the new task is being plotted.

    Returns:
    tuple: A pair of arrays representing the bins for the histograms.
    """
    # Selecting different bin ranges based on whether the new task is being plotted
    if plot_newtask:
        binsA = np.arange(-5, 125, 10)  # Bins for plotting the new task
        binsB = np.arange(-5, 125, 14)
    else:
        binsA = np.arange(-5, 105, 10)  # Bins for plotting familiar tasks
        binsB = np.arange(-5, 105, 14)
    return binsA, binsB


def fig4b_plot(ax, famtask1_prog, famtask2_prog, bins, 
               famtask1_example_trial, 
               famtask2_example_trial, pval, map_type,
               plot_newtask,
               newtask_prog=None):
    """
    Plot histogram and scatter plot for familiar and new tasks on a given axis.

    Args:
    ax (matplotlib.axes.Axes): The axis to plot on.
    famtask1_prog, famtask2_prog (array): Progress data for the familiar tasks.
    bins (array): Bins for histogram plotting.
    famtask1_example_trial, famtask2_example_trial (array): Example trial data for the familiar tasks.
    pval (float): P-value for statistical significance.
    map_type (str): Type of the map, used to determine color settings.
    plot_newtask (bool): Flag to indicate whether to plot the new task.
    newtask_prog (array, optional): Progress data for the new task. Defaults to None.
    """
    # Check if the data fits within the specified bins
    fig4b_check_bins(bins, famtask1_prog, famtask2_prog, plot_newtask, newtask_prog=newtask_prog)
    
    # Plot histograms for the familiar tasks
    ax.hist(famtask2_prog, histtype='step', bins=bins, color=task_to_color('FamTask2'))
    ax.hist(famtask1_prog, histtype='step', bins=bins, color=task_to_color('FamTask1'))

    # If plotting new task, plot its histogram as well
    if plot_newtask: 
        ax.hist(newtask_prog, histtype='step', bins=bins, color=task_to_color('NewTask'))
    
    # Set a fixed height above the histograms to place scatter plots and statistical significance markers
    above_hist_height = 13
    
    # Plotting mean markers for the progress data of each task
    ax.scatter(np.mean(famtask2_prog), [above_hist_height], marker='v', color=task_to_color('FamTask2'), s=4)
    ax.scatter(np.mean(famtask1_prog), [above_hist_height], marker='v', color=task_to_color('FamTask1'), s=4)
    
    # Plot a bar with stars to denote statistical significance
    plot_bar_with_stars(np.mean(famtask2_prog), left_dist_mean=np.mean(famtask1_prog),
                          top_data_lim=above_hist_height + 1,
                          pval=pval, ax=ax, linewidth=1)

    # If plotting new task, plot its mean marker as well
    if plot_newtask:
        ax.scatter(np.mean(newtask_prog), [above_hist_height], marker='v', color=task_to_color('NewTask'), s=4)

    # If not plotting new task, plot example trials for familiar tasks
    if not plot_newtask:
        fig4b_plot_example_trials(ax, famtask1_example_trial, famtask2_example_trial)
    
    # Apply standard formatting to the plot
    fig4b_set_plot_formatting(ax, map_type)


###############################
###### Arrays/Smoothing #######
###############################
def center_smooth(x, y, window_size):
    """
    Smooths and centers a dataset given a smoothing window size, and optionally applies a window function.
    
    Parameters:
    - x (np.ndarray or list): The x-values of the dataset.
    - y (np.ndarray or list): The y-values of the dataset to be smoothed.
    - window_size (int): The size of the smoothing window. Must be an even number.
    
    Returns:
    - tuple: A pair `(x_adjusted, y_smoothed)` where `x_adjusted` are the x-values adjusted to account for the centering effect of the smoothing, and `y_smoothed` are the smoothed y-values.
    
    Notes:
    - The smoothing is applied using the `causal_smooth` function, which supports 'box' or 'triangle' window functions.
    - This function asserts that the window size is even to ensure proper centering.
    - `x_adjusted` is computed by subtracting half the window size from the original x-values to center the effect of the smoothing operation.
    """
    assert window_size % 2 == 0
    half_window = int(window_size / 2)
    y_smooth = causal_smooth(y, window_size, nan_start=False)
    return x - half_window, y_smooth

def causal_smooth(x, window_size, window_function='box', nan_start=False, trim_front=0):
    """
    Smooth a 1D array or list using a specified window function.

    Args:
        x (list or np.ndarray): Input data to be smoothed, must be 1-dimensional.
        window_size (int): The size of the smoothing window. Must be a non-negative integer.
        window_function (str): Type of window function ('box' or 'triangle'). Defaults to 'box'.
        nan_start (bool): If True, ignores the first `window_size` elements for smoothing. Defaults to False.
        trim_front (int): Number of elements at the start of the result to set as NaN. Must be non-negative. Defaults to 0.

    Returns:
        np.ndarray: Smoothed array.

    Raises:
        ValueError: If `x` is not a 1D array or list, or if `window_size`, `window_function`, or `trim_front` are invalid.

    Example:
        >>> causal_smooth([1, 2, 3, 4, 5], 3, 'triangle')
        array([nan, nan, 2., 3., 4.])
    """

    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("Input 'x' must be a list or numpy array.")
    if not isinstance(window_size, int) or window_size < 0:
        raise ValueError("Window size must be a non-negative integer.")
    if window_size == 0:
        return np.array(x)
    if window_function not in ['box', 'triangle']:
        raise ValueError("Window function must be 'box' or 'triangle'.")
    if not isinstance(nan_start, bool):
        raise ValueError("'nan_start' must be a boolean value.")
    if not isinstance(trim_front, int) or trim_front < 0:
        raise ValueError("'trim_front' must be a non-negative integer.")

    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("Input must be 1-dimensional.")

    # Prepare the window weights
    if window_function == 'triangle':
        n = np.arange(1, window_size + 1)
        w = 1 - (window_size - n) / window_size
    else:  # box
        w = np.ones(window_size)

    # Smooth the data
    v_smoothed = np.full_like(x, fill_value=np.nan, dtype=np.float64)
    start = window_size if nan_start else 0

    for i in range(start, len(x)):
        window = x[max(i - window_size, 0): i]
        is_finite = np.isfinite(window)
        w_window = w[:len(window)]

        if is_finite.any():
            v_smoothed[i] = np.nansum(w_window[is_finite] * window[is_finite]) / np.nansum(w_window[is_finite])

    if trim_front > 0:
        v_smoothed[:trim_front] = np.nan
    return v_smoothed


#################################
####### Tuning Curves ###########
#################################
def fit_cosine_tuning_curve(spike_counts, rad_angles):
    """ λ(s) = $$r_0 + (r max − r_0 ) cos(s − s_{max} )$$
        r is the spiking rates
        s = theta in my code below
        λ is the spike counts.

        spike counts as a function of angle s = baseline_firing rate + amplitude cosine(s - smax)

    Terms:
        max_fr = rmax
        baseline_fr = r0
        predicted_spike_counts = lmbda
    """
    # Spike Counts: Angles x Spikes
    assert_in_radian_range(rad_angles)
    num_conditions = spike_counts.shape[0]
    if np.any(rad_angles > 2 * np.pi): raise ValueError('Angles May be in Degrees')
    if np.any(rad_angles < 0): warnings.warn('Negative Angles')
    lmbda = spike_counts
    assert lmbda.shape == (num_conditions,)
    A = np.vstack([np.ones(num_conditions),
                   np.cos(rad_angles),
                   np.sin(rad_angles)]).T
    assert np.all(A[:, 0] == np.ones(num_conditions)) and (np.all(A[:, 1] == np.cos(rad_angles)))
    x, _, _, _ = np.linalg.lstsq(A, lmbda, rcond=None)
    a = x[1]
    b = x[2]

    try:    theta_max_rad = np.arctan(b / a)
    except:
        theta_max_rad = np.nan; print(f'{x[2], x[1]}')
        raise
    a_coeff = a / np.cos(theta_max_rad)
    b_coeff = b / np.sin(theta_max_rad)
    try:
        np.testing.assert_allclose(a_coeff, b_coeff, atol=0.000000001)
    except:
        print('Assertion error', a_coeff, b_coeff)
  
    r0 = x[0]
    rmax = r0 + x[1] / np.cos(theta_max_rad)
    if r0 > rmax:
        rmin = rmax
        rmax = 2 * r0 - rmin
        theta_max_rad = theta_max_rad + np.pi
    theta_max_rad = theta_max_rad % (2 * np.pi)

    """ s_max = atan(x(3)/x(2)) """
    assert 0 <= theta_max_rad <= 2 * np.pi, theta_max_rad
    assert rmax > 0
    pref_dir = theta_max_rad
    max_fr = rmax
    baseline_fr = r0
    predicted_spike_counts = lmbda
    return baseline_fr, max_fr, pref_dir, predicted_spike_counts

def cosine_tune_fn(angles, baseline_fr, max_fr, pref_dir):
    """ λ(s) = r 0 + (r max − r 0 ) cos(s − s max)
    s is the reaching angle of the arm (in degrees),
    smax (smax in radians) is the reaching angle associated with the maximum response rmax
    r0 is an offset that shifts the tuning curve up from the zero axis
        returns  lambda =     predicted_spike_counts

    Angles must be in radians.
    """
    s = angles
    smax = pref_dir
    rmax = max_fr
    r0 = baseline_fr
    assert_in_radian_range(s)
    predicted_spike_counts = r0 + (rmax - r0) * np.cos(s - smax) # lambda(s)
    return predicted_spike_counts